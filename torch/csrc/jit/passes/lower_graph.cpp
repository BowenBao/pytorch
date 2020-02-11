#include <torch/csrc/jit/passes/lower_graph.h>
#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/script/error_report.h>
#include <unordered_map>

namespace torch {
namespace jit {

struct Slot {
  c10::intrusive_ptr<c10::ivalue::Object> obj;
  size_t offset;
  bool operator==(const Slot& other) const {
    return (this->obj == other.obj && this->offset == other.offset);
  }
};

struct SlotValue {
  enum SlotValueType {
    Initializer,
    Value,
  };
  SlotValueType type;
  size_t offset;
  torch::jit::Value* value;
};

// remove the first module argument, replacing any access of its
// parameters/attributes with extra_ivalue input Slots that hold what value to
// pass into the graph. Used for ONNX export to remove first-class modules
// so it can deal purely with parameters and inputs
std::pair<std::shared_ptr<Graph>, std::vector<Slot>> lower_graph(
    const ModulePtr& self,
    Graph& g_,
    size_t self_offset = 0) {
  std::shared_ptr<Graph> g = g_.copy();
  // Inline to remove method/function calls
  Inline(*g);

  std::vector<Slot> extra_ivalues;

  struct SlotHash {
    std::size_t operator()(const Slot& slot) const {
      auto obj_hash = std::hash<c10::ivalue::Object*>{}(slot.obj.get());
      auto offset_hash = std::hash<size_t>{}(slot.offset);
      return torch::hash_combine(obj_hash, offset_hash);
    }
  };
  std::unordered_map<Slot, SlotValue, SlotHash> slot_to_value;
  struct ToScan {
    ModulePtr mod;
    Node* n;
    size_t offset;
  };
  std::vector<ToScan> to_scan;
  std::vector<Node*> to_clean; // nodes that should be dead at the end

  auto getOrAddSlot = [&](const Slot& slot) -> Value* {
    auto it = slot_to_value.find(slot);
    if (it != slot_to_value.end()) {
      auto slot_value = it->second;
      switch (slot_value.type) {
        case SlotValue::SlotValueType::Initializer: {
          size_t ivalues_start = g->inputs().size() - extra_ivalues.size();
          return g->inputs().at(ivalues_start + slot_value.offset);
        }
        case SlotValue::SlotValueType::Value:
          return slot_value.value;
      }
    }

    auto iv = slot.obj->getSlot(slot.offset);
    if (!iv.isTensor()) {
      WithInsertPoint guard(*g->nodes().begin());
      auto v = g->insertConstant(iv);
      slot_to_value[slot] = {SlotValue::SlotValueType::Value, 0, v};
      return v;
    } else {
      extra_ivalues.emplace_back(slot);
      slot_to_value[slot] = {SlotValue::SlotValueType::Initializer, extra_ivalues.size() - 1, nullptr};
      return g->addInput()->setType(slot.obj->getSlot(slot.offset).type());
    }
  };

  auto self_value = g->inputs().at(self_offset);

  for (int i = self_value->uses().size() - 1; i >= 0; --i) {
    Use use = self_value->uses()[i];
    to_scan.emplace_back(ToScan{self, use.user, use.offset});
  }
  // for (Use use : self_value->uses()) {
  //   // printf("Emplacing module %s node %s (%zu)outputs and offset %zu\n",
  //   //   self->name().c_str(),
  //   //   use.user->output(0)->type()->python_str().c_str(),
  //   //   use.user->outputs().size(),
  //   //   use.offset);
  //   to_scan.emplace_back(ToScan{self, use.user, use.offset});
  // }
  while (to_scan.size() > 0) {
    auto e = to_scan.back();
    to_scan.pop_back();

    // when we lambda lift forks, first-class modules may be passed across
    // forks. This code recursively lowers the module in the fork call.
    if (e.n->kind() == prim::fork) {
      auto subgraph = e.n->g(attr::Subgraph);
      std::vector<Slot> new_slots;
      std::tie(subgraph, new_slots) = lower_graph(e.mod, *subgraph, e.offset);
      e.n->g_(attr::Subgraph, subgraph);
      for (const Slot& slot : new_slots) {
        e.n->addInput(getOrAddSlot(slot));
      }
      e.n->removeInput(e.offset);
      continue;
    }
    if (e.n->kind() == prim::PythonOp) {
      throw script::ErrorReport(e.n->sourceRange())
          << "Couldn't export Python method.";
    }
    if (e.n->kind() == prim::SetAttr) {
      size_t slot_idx = e.mod->type()->getAttributeSlot(e.n->s(attr::name));
      AT_ASSERT(e.n->inputs().size() >= 2);
      slot_to_value[{e.mod, slot_idx}] = {SlotValue::SlotValueType::Value, 0, e.n->input(1)};
      e.n->destroy();
      continue;
    }
    if (e.n->kind() != prim::GetAttr) {
      throw script::ErrorReport(e.n->sourceRange())
          << "temporary: the only valid use of a module is looking up an "
             "attribute but found "
          << *e.n;
    }
    // printf("kind is %s\n", e.n->kind().toDisplayString()); // prim::getAttr
    size_t slot_idx = e.mod->type()->getAttributeSlot(e.n->s(attr::name));
    // printf("output type is %s (%zu)outputs with attr name %s slot_idx: %zu of module %s\n",
    //   e.n->output(0)->type()->python_str().c_str(),
    //   e.n->outputs().size(),
    //   e.n->s(attr::name).c_str(),
    //   slot_idx,
    //   e.mod->name().c_str());
    // printf("", e.mod->getSlot(slot_idx));
    auto iv = e.mod->getSlot(slot_idx);
    if (ClassTypePtr c = e.n->output()->type()->cast<ClassType>()) {
      if (c->is_module()) {
        for (int i = e.n->output()->uses().size() - 1; i >= 0; --i) {
          Use use = e.n->output()->uses()[i];
          to_scan.emplace_back(ToScan{iv.toObject(), use.user, use.offset});
        }
        // for (Use use : e.n->output()->uses()) {
        //   printf("Emplacing module %s node %s and offset %zu\n",
        //     iv.toObject()->name().c_str(),
        //     use.user->kind().toDisplayString(),
        //     use.offset);
        //   to_scan.emplace_back(ToScan{iv.toObject(), use.user, use.offset});
        // }
        to_clean.emplace_back(e.n);
        continue;
      }
    }

    e.n->output()->replaceAllUsesWith(getOrAddSlot({e.mod, slot_idx}));
    e.n->destroy();
  }

  while (to_clean.size() > 0) {
    Node* n = to_clean.back();
    AT_ASSERT(!n->hasUses());
    n->destroy();
    to_clean.pop_back();
  }
  AT_ASSERT(!self_value->hasUses());
  g->eraseInput(self_offset);

  return std::make_pair(std::move(g), std::move(extra_ivalues));
}

static std::vector<at::Tensor> loadTensors(const std::vector<Slot>& slots) {
  std::vector<at::Tensor> result;
  result.reserve(slots.size());
  for (const Slot& slot : slots) {
    result.emplace_back(slot.obj->getSlot(slot.offset).toTensor());
  }
  return result;
}

std::pair<std::shared_ptr<Graph>, std::vector<at::Tensor>> LowerGraph(
    Graph& graph,
    const ModulePtr& self) {
  auto result = lower_graph(self, graph);
  return std::make_pair(result.first, loadTensors(result.second));
}

} // namespace jit
} // namespace torch
