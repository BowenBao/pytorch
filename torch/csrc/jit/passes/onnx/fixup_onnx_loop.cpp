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

Node* CreateEmptyConstantNode(Graph* graph) {
  Node* node = graph->create(onnx::Constant);
  node->t_(
    attr::value,
    autograd::make_variable(at::empty(
      {0},
      at::kLong)));
  return node;
}

Node* CreateReshapeNode(Value* inp, Value* shape, Graph* graph) {
  Node* node = graph->create(onnx::Reshape);
  node->addInput(inp);
  node->addInput(shape);
  return node;
}

Node* InsertCastAndReshapeForCond(Value* cond_val, Graph* graph, Node* consumer_node) {
  // prev:  cond_val -> consumer_node
  // after: cond_val -> cast -> reshape -> consumer_node
  Node* cast_node = CreateCastToBoolNode(cond_val, graph);
  cast_node->insertBefore(consumer_node);

  Node* empty_constant_node = CreateEmptyConstantNode(graph);
  empty_constant_node->insertAfter(cast_node);

  Node* reshape_node = CreateReshapeNode(cast_node->output(), empty_constant_node->output(), graph);
  reshape_node->insertAfter(empty_constant_node);

  consumer_node->replaceInputWith(cond_val, reshape_node->output());

  return reshape_node;
}

void FixupONNXLoops(Block* block) {
  for (auto* node : block->nodes()) {
    if (node->kind() == ::c10::onnx::Loop) {
      auto* loop_node = node;
      auto* graph = loop_node->owningGraph();

      // add cast & reshape to condition input outside the loop.
      Value* cond_val = loop_node->inputs()[1];
      InsertCastAndReshapeForCond(cond_val, graph, loop_node);

      // Setup Loop input cond and i.
      AT_ASSERT(loop_node->blocks().size() == 1);
      auto* sub_block = loop_node->blocks()[0];
      Value* cond = sub_block->insertInput(1, "cond");
      cond->setType(CompleteTensorType::create(at::kBool, at::kCPU, {}));
      Value* i = sub_block->inputs()[0];
      i->setType(CompleteTensorType::fromNumberType(IntType::get()));

      // add cast & reshape to condition input inside the loop.
      Value* next_cond_val = sub_block->outputs()[0];
      InsertCastAndReshapeForCond(next_cond_val, graph, sub_block->return_node());
    }
    for (Block* block : node->blocks()) {
      FixupONNXLoops(block);
    }
  }
}

void FixupONNXLoops(std::shared_ptr<Graph>& graph) {
  FixupONNXLoops(graph->block());
}

} // namespace jit
} // namespace torch
