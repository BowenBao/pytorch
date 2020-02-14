#include <torch/csrc/jit/passes/onnx/fixup_onnx_conditionals.h>

namespace torch {
namespace jit {

namespace onnx{
using namespace ::c10::onnx;
}

void FixupONNXIfs(Block* block) {
  for (auto* node : block->nodes()) {
    if (node->kind() == ::c10::onnx::If) {
      auto* if_node = node;
      auto* graph = if_node->owningGraph();
      for (Block* block : node->blocks()) {
        FixupONNXIfs(block);
        if (block->nodes().begin() == block->nodes().end()) {
          //ONNX does not support empty blocks, must use some op which does nothing
          Value* output = block->outputs()[0];
          Node* id_node = graph->create(onnx::Identity);
          id_node->insertBefore(block->return_node());
          id_node->addInput(output);
          id_node->output()->copyMetadata(output);
          block->return_node()->replaceInputWith(output, id_node->output());
        }

        for (size_t i = 0; i < block->outputs().size(); ++i) {
          auto block_out = block->outputs()[i];
          auto node_out = if_node->outputs()[i];

          if (!block_out->type()->cast<TensorType>()->sizes().isComplete() &&
              node_out->type()->cast<TensorType>()->sizes().isComplete()) {
            block_out->copyMetadata(node_out);
          }
        }
      }
    }
    else {
      for (Block* block : node->blocks()) {
        FixupONNXIfs(block);
      }
    }
  }
}

void FixupONNXConditionals(std::shared_ptr<Graph>& graph) {
  FixupONNXIfs(graph->block());
}

} //jit
} //torch
