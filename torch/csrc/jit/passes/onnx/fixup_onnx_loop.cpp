#include <torch/csrc/jit/passes/onnx/fixup_onnx_loop.h>

namespace torch {
namespace jit {

namespace onnx{
using namespace ::c10::onnx;
}

Node* CreateCastToBoolNode(Value* val, Graph* graph) {
  Node* cast_node = graph->create(onnx::Cast);
  cast_node->addInput(val);
  cast_node->i_(attr::to, /*Bool*/9);
  return cast_node;
}

Node* InsertCastForCond(Value* cond_val, Graph* graph, Node* consumer_node) {
  // prev:  cond_val -> consumer_node
  // after: cond_val -> cast -> consumer_node
  // NOTE: The cast is required because operators like PyTorch Greater/Less
  //       return tensor in type torch.uint8. However the type for condition
  //       input in ONNX Loop must be bool.
  Node* cast_node = CreateCastToBoolNode(cond_val, graph);
  cast_node->insertBefore(consumer_node);

  consumer_node->replaceInputWith(cond_val, cast_node->output());
  return cast_node;
}

bool IsCondCastRequired(Value* cond_val) {
  const auto& type = cond_val->type();
  if (auto tt = type->cast<TensorType>()) {
    if (auto scalar_type = tt->scalarType()) {
      return *scalar_type != c10::kBool;
    }
  }
  return !type->isSubtypeOf(BoolType::get());
}

void FixupONNXLoops(Block* block) {
  for (auto* node : block->nodes()) {
    if (node->kind() == ::c10::onnx::Loop) {
      auto* loop_node = node;
      auto* graph = loop_node->owningGraph();

      // add cast to condition input outside the loop.
      Value* cond_val = loop_node->inputs()[1];
      if (IsCondCastRequired(cond_val))
        InsertCastForCond(cond_val, graph, loop_node);

      // Setup Loop input cond and i.
      TORCH_INTERNAL_ASSERT(loop_node->blocks().size() == 1);
      auto* sub_block = loop_node->blocks()[0];
      Value* cond = sub_block->insertInput(1, "cond");
      cond->setType(BoolType::create());

      Value* i = sub_block->inputs()[0];
      i->setType(TensorType::fromNumberType(IntType::get()));

      // add cast to condition input inside the loop.
      Value* next_cond_val = sub_block->outputs()[0];
      if (IsCondCastRequired(next_cond_val))
        InsertCastForCond(next_cond_val, graph, sub_block->return_node());
    }
    for (Block* block : node->blocks()) {
      FixupONNXLoops(block);
    }
  }
}

namespace {
bool IsErasableSequence(const Node* loop_node, size_t i) {
  AT_ASSERT(loop_node->blocks().size() == 1);
  auto* sub_block = loop_node->blocks()[0];
  auto* out_node = sub_block->outputs()[i-1]->node();
  auto* in_val = sub_block->inputs()[i];

  if (out_node->kind() != ::c10::onnx::SequenceInsert) {
    return false;
  }

  if (out_node->inputs().size() == 3) {
    // Non-default insert position is not supported.
    return false;
  }

  if (out_node->input(0) != in_val) {
    // Only SequenceInsert that applies on loop-carried sequence is supported.
    return false;
  }

  if (loop_node->inputs()[i]->node()->kind() != ::c10::onnx::SequenceConstruct) {
    // Initial sequence must be empty.
    return false;
  }

  if (out_node->output()->uses().size() != 1) {
    // The sequence is not supported to be used elsewhere.
    return false;
  }

  return true;
}
} // anonymous namespace

// ONNX::Loop does not support Sequence type as loop-carried dependencies. Only tensors are supported.
// This pass converts Sequence loop-carried dependencies to scan_outputs.
// In opset 11, only the below pattern is supported.
//
// PTIR graph:
//  ...
//  %res.1 : Tensor[] = prim::ListConstruct()
//  %res : Tensor[] = prim::Loop(%11, %22, %res.1)
//    block0(%i.1 : Tensor, %res.6 : Tensor[]):
//      ...
//      %res.3 : Tensor[] = aten::append(%res.6, %17)
//      -> (%22, %res.3)
//  return (%res.3)
//
// ONNX graph:
//  ...
//  %res.1 : Tensor[] = onnx::SequenceEmpty()
//  %res : Tensor = onnx::Loop(%11, %22, %res.1)
//    block0(%i.1 : Tensor):
//      ...
//      -> (%22, %17)
//  %res_seq : Tensor[] = onnx::SplitToSequence[keepdims=0](%res)
//  return (%res_seq)
void ConvertSequenceDependencies(Block* block) {
  for (auto* node : block->nodes()) {
    for (Block* block : node->blocks()) {
      ConvertSequenceDependencies(block);
    }

    if (node->kind() == ::c10::onnx::Loop) {
      auto* loop_node = node;
      auto* graph = loop_node->owningGraph();

      AT_ASSERT(loop_node->blocks().size() == 1);
      auto* sub_block = loop_node->blocks()[0];

      // loop sub-block inputs are (iter, cond, loop-carried dependencies)
      // loop sub-block outputs are (cond, loop-carried dependencies, scan outputs)
      // loop inputs are (iter, cond, loop-carried dependencies)
      // loop outputs are (loop-carried dependencies, scan outputs)
      for (size_t i = 2; i < sub_block->inputs().size(); ++i) {
        // TODO: more condition checks. We need to ensure there are no other uses for the node. We might be able to extend this condition, but not too much, since sequence cannot be updated really.
        printf("sub block output kind is %s\nsub block output node's input kind is %s\nsub block input is %s\nloop input kind is %s\nsub block input count %d\n",
            sub_block->outputs()[i-1]->node()->kind().toDisplayString(),
            sub_block->outputs()[i-1]->node()->input(0)->debugName().c_str(),
            sub_block->inputs()[i]->debugName().c_str(),
            loop_node->inputs()[i]->node()->kind().toDisplayString(),
            int(sub_block->inputs().size()));
        if (IsErasableSequence(loop_node, i)) {


          auto* out_node = sub_block->outputs()[i-1]->node();
          sub_block->return_node()->replaceInputWith(out_node->output(), out_node->input(1));

          Node* split_node = loop_node->owningGraph()->create(onnx::SplitToSequence);
          loop_node->outputs()[i-2]->replaceAllUsesWith(split_node->output());
          split_node->i_(attr::keepdims, 0);
          split_node->addInput(loop_node->outputs()[i-2]);
          split_node->insertAfter(loop_node);

          split_node->output()->copyMetadata(loop_node->outputs()[i-2]);
          // TODO: shape is not correct here. deal with shape later.
          loop_node->outputs()[i-2]->copyMetadata(out_node->input(1));
          

          out_node->destroy();
          sub_block->eraseInput(i);
          loop_node->removeInput(i);
        }
      }
    }

  }
}

void FixupONNXLoops(std::shared_ptr<Graph>& graph) {
  FixupONNXLoops(graph->block());
  ConvertSequenceDependencies(graph->block());
}

} // namespace jit
} // namespace torch
