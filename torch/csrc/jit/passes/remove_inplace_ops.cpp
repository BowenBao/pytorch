#include <torch/csrc/jit/passes/remove_inplace_ops.h>

namespace torch {
namespace jit {
namespace {
static const std::unordered_map<NodeKind, NodeKind> inPlaceToOutOfPlace = {
    {aten::add_, aten::add},
    {aten::sub_, aten::sub},
    {aten::div_, aten::div},
    {aten::mul_, aten::mul}};

bool isInplaceOp(const Node* node) {
  return inPlaceToOutOfPlace.count(node->kind()) != 0;
}

// Remove all in-place ops and replace them with out-of-place equivalents.
// e.g.
//   %foo = aten::add_(%foo, %n)
// becomes
//   %foo.2 = aten::add(%foo, %n)
//
// NOTE: this is NOT SAFE, since it assumes that the LHS is not aliased by
// another value. This is only to avoid breaking ONNX export; when alias
// analysis is done we can emit a warning if someone tries to export.
void RemoveInplaceOps(Block* block) {
  auto graph = block->owningGraph();
  auto it = block->nodes().begin();
  while (it != block->nodes().end()) {
    auto node = *it;
    ++it;
    for (auto block : node->blocks()) {
      RemoveInplaceOps(block);
    }

    if (isInplaceOp(node)) {
      // create a replacement out of place op
      auto newNode = graph->create(inPlaceToOutOfPlace.at(node->kind()));
      newNode->insertBefore(node);
      newNode->setScope(node->scope());
      // copy inputs
      for (auto input : node->inputs()) {
        newNode->addInput(input);
      }

      // Create a new output node and replace all uses of self with it
      newNode->output()->copyMetadata(node->output());
      node->replaceAllUsesWith(newNode);
      node->destroy();
    }
  }
}

void preprocessListPop(Block* b) {
  for (auto it = b->nodes().begin(), end = b->nodes().end(); it != end; ++it) {
    for (auto* child_block : it->blocks()) {
      preprocessListPop(child_block);
    }

    if (it->kind() == aten::pop) {
      //   %ten : Tensor = aten::pop(%seq, %pos)
      // Convert to
      //   %ten : Tensor = aten::__getitem__(%seq, %pos)
      //   %new_seq : Tensor[] = aten::pop(%seq, %pos)
      // And replace all uses of %seq afterwards with %new_seq
      Node* getitem_node =
          b->owningGraph()->create(aten::__getitem__, {it->inputs()});
      getitem_node->output()->copyMetadata(it->output());
      getitem_node->insertBefore(*it);
      it->output()->replaceAllUsesWith(getitem_node->output());

      it->output()->copyMetadata(it->inputs()[0]);
      it->inputs()[0]->replaceAllUsesAfterNodeWith(*it, it->output());
    }
  }
}
} // namespace

void RemoveInplaceOps(const std::shared_ptr<Graph>& graph) {
  RemoveInplaceOps(graph->block());
  preprocessListPop(graph->block());
}
} // namespace jit
} // namespace torch
