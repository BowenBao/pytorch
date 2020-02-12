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
  // enum SlotValueType {
  //   Initializer,
  //   Value,
  // };
  // SlotValueType type;
  size_t offset;
  std::vector<torch::jit::Value*> values;
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
  std::unordered_map<Node*, std::unordered_map<Slot, size_t, SlotHash>> node_to_slot_output;
  struct ToScan {
    ModulePtr mod;
    Node* n;
    size_t offset;
  };
  std::vector<ToScan> to_scan;
  std::vector<Node*> to_clean; // nodes that should be dead at the end

  auto isSameOrParentBlock = [&](const Block* parent, const Block* b) -> bool {
    auto* child = b;
    while (child) {
      if (parent == child) return true;
      if (nullptr == child->owningNode()) return false;
      child = child->owningNode()->owningBlock();
    }
    return false;
  };

  auto updateSlotValues = [&](const Slot& slot, std::vector<Value*>& vs) {
    auto it = slot_to_value.find(slot);
    AT_ASSERT(it != slot_to_value.end());
    auto slot_value = it->second;
    for (auto v : vs) {
      for (int i = 0; i < vs.size(); ++i) {
        auto exist_v = vs[i];
        if (exist_v->node()->owningBlock() == v->node()->owningBlock()) {
          vs[i] = v;
          break;
        }
      }
      vs.emplace_back(v);
    }
  };

  auto getOrAddSlot = [&](const Slot& slot, const Node* node) -> Value* {
    auto it = slot_to_value.find(slot);
    if (it != slot_to_value.end()) {
      auto slot_value = it->second;
      if (slot_value.values.size() != 0) {
        for (auto v : slot_value.values) {
          if (isSameOrParentBlock(v->node()->owningBlock(), node->owningBlock())
            && v->node()->isBefore(node)) {
            return v;
          }
        }
      }
      AT_ASSERT(slot_value.offset != 0);
      size_t ivalues_start = g->inputs().size() - extra_ivalues.size();
      return g->inputs().at(ivalues_start + slot_value.offset);
      // switch (slot_value.type) {
      //   case SlotValue::SlotValueType::Initializer: {
      //     size_t ivalues_start = g->inputs().size() - extra_ivalues.size();
      //     return g->inputs().at(ivalues_start + slot_value.offset);
      //   }
      //   case SlotValue::SlotValueType::Value:
      //     for (auto v : slot_value.values) {
      //       if (v->node()->owningBlock() == ) {
      //         return v;
      //       }
      //     }
      //     return slot_value.value;
      // }
    }

    auto iv = slot.obj->getSlot(slot.offset);
    if (!iv.isTensor()) {
      WithInsertPoint guard(*g->nodes().begin());
      auto v = g->insertConstant(iv);
      slot_to_value[slot] = {0, {v}};
      return v;
    } else {
      extra_ivalues.emplace_back(slot);
      slot_to_value[slot] = {extra_ivalues.size() - 1, {}};
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
    printf("To scan node kind %s\n", e.n->kind().toDisplayString());

    // when we lambda lift forks, first-class modules may be passed across
    // forks. This code recursively lowers the module in the fork call.
    if (e.n->kind() == prim::fork) {
      auto subgraph = e.n->g(attr::Subgraph);
      std::vector<Slot> new_slots;
      std::tie(subgraph, new_slots) = lower_graph(e.mod, *subgraph, e.offset);
      e.n->g_(attr::Subgraph, subgraph);
      for (const Slot& slot : new_slots) {
        e.n->addInput(getOrAddSlot(slot, e.n));
      }
      e.n->removeInput(e.offset);
      continue;
    }
    if (e.n->kind() == prim::PythonOp) {
      throw script::ErrorReport(e.n->sourceRange())
          << "Couldn't export Python method.";
    }
    if (e.n->kind() == prim::SetAttr) {
      printf("Enter SetAttr\n");
      size_t slot_idx = e.mod->type()->getAttributeSlot(e.n->s(attr::name));
      printf("SetAttr %s on slot %zu\n", e.n->s(attr::name).c_str(), slot_idx);
      AT_ASSERT(e.n->inputs().size() >= 2);
      std::vector<torch::jit::Value*> vs = {e.n->input(1)};
      Slot slot = {e.mod, slot_idx};

      auto owning_block = e.n->owningBlock();
      auto owning_node = owning_block->owningNode();
      if (owning_node && owning_node->kind() == prim::If) {
        // Update SetAttr for output of If node.
        printf("Set Attr owning node is prim::If.\n");
        bool is_output_exist = false;
        if (node_to_slot_output.find(owning_node) != node_to_slot_output.end()) {
          auto& slot_to_output = node_to_slot_output[owning_node];
          if (slot_to_output.find(slot) != slot_to_output.end()) {
            printf("Set attr found existing output\n");
            owning_block->return_node()->replaceInputWith(owning_block->outputs()[slot_to_output[slot]], e.n->input(1));
            is_output_exist = true;
          }
        }
        if (!is_output_exist) {
          printf("Set attr adding new output\n");
          node_to_slot_output[owning_node][slot] = owning_block->outputs().size();
          owning_block->return_node()->addInput(e.n->input(1));
          auto else_block = owning_node->blocks()[1];
          if (else_block != owning_block) {
            // TODO: cannot add node for else branch. This breaks all the offset and uses and to_scan list.
            // Need to handle it separately.

            // auto get_attr_node = g->create(prim::GetAttr, {e.n->input(0)}, 1);
            // get_attr_node->copyAttributes(*e.n);
            // else_block->appendNode(get_attr_node);
            // else_block->return_node()->addInput(get_attr_node->output());
          }
          vs.emplace_back(owning_node->addOutput());
        }
      }
      printf("now graph looks like: %s\n", g->toString().c_str());
      if (slot_to_value.find(slot) != slot_to_value.end()) {
        printf("Set attr update slot\n");
        updateSlotValues(slot, vs);
      } else {
        printf("Set attr new slot\n");
        slot_to_value[{e.mod, slot_idx}] = {0, vs};
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
    printf("GetAttr %s on slot %zu\n", e.n->s(attr::name).c_str(), slot_idx);
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

    e.n->output()->replaceAllUsesWith(getOrAddSlot({e.mod, slot_idx}, e.n));
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
