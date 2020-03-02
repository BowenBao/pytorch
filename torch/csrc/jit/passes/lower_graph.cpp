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
  size_t offset;
  torch::jit::Value* value = nullptr;
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
      if (nullptr == slot_value.value) {
        size_t ivalues_start = g->inputs().size() - extra_ivalues.size();
        return g->inputs().at(ivalues_start + slot_value.offset);
      } else {
        return slot_value.value;
      }
    }

    auto iv = slot.obj->getSlot(slot.offset);
    if (!iv.isTensor()) {
      // printf("iv kind is %s\n", iv.type()->python_str().c_str());
      if (iv.isTensorList()) {
        auto ts = iv.toTensorList();
        std::vector<torch::jit::Value*> vs;
        for (const at::Tensor& t : ts) {
          vs.emplace_back(g->addInput()->setType(TensorType::create(t)));
        }
        WithInsertPoint guard(*g->nodes().begin());
        auto v = g->insertNode(g->create(prim::ListConstruct, vs))->output();
        v->setType(iv.type());
        slot_to_value[slot] = {0, v};
        extra_ivalues.emplace_back(slot);
        return v;
      } else {
        WithInsertPoint guard(*g->nodes().begin());
        auto v = g->insertConstant(iv);
        slot_to_value[slot] = {0, v};
        return v;
      }
    } else {
      extra_ivalues.emplace_back(slot);
      slot_to_value[slot] = {extra_ivalues.size() - 1, nullptr};
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
    // printf("To scan node kind %s\n", e.n->kind().toDisplayString());

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
      // printf("Enter SetAttr\n");
      size_t slot_idx = e.mod->type()->getAttributeSlot(e.n->s(attr::name));
      // printf("SetAttr %s on slot %zu\n", e.n->s(attr::name).c_str(), slot_idx);
      AT_ASSERT(e.n->inputs().size() >= 2);
      auto v = e.n->input(1);
      Slot slot = {e.mod, slot_idx};

      auto owning_block = e.n->owningBlock();
      auto owning_node = owning_block->owningNode();
      if (owning_node && owning_node->kind() == prim::If) {
        std::cerr << "Warning: SetAttr within prim::If is currently not supported. "
                  << "The exported graph may be different." << std::endl;
        e.n->destroy();
        continue;
      }
      // printf("now graph looks like: %s\n", g->toString().c_str());
      if (slot_to_value.find(slot) != slot_to_value.end()) {
        // printf("Set attr update slot\n");
        slot_to_value[slot].value = v;
      } else {
        // printf("Set attr new slot\n");
        slot_to_value[slot] = {0, v};
      }
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
    // printf("GetAttr %s on slot %zu\n", e.n->s(attr::name).c_str(), slot_idx);
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
  // printf("loadTensors: %zu slots.\n", slots.size());
  for (const Slot& slot : slots) {
    auto iv = slot.obj->getSlot(slot.offset);
    if (iv.isTensorList()) {
      // printf("loadTensors: tensor list of %zu tensors.\n", iv.toTensorList().size());
      for (const at::Tensor& v : iv.toTensorList()) {
        result.emplace_back(v);
      }
    } else {
      result.emplace_back(iv.toTensor());
    }
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
