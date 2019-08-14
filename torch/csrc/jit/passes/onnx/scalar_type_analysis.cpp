#include <torch/csrc/jit/passes/onnx/scalar_type_analysis.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>

namespace torch {
namespace jit {

namespace onnx {
using namespace ::c10::onnx;
}

namespace {
class ScalarTypeHashFunction {
 public:
  size_t operator()(const c10::ScalarType& type) const {
    return static_cast<size_t>(type);
  }
};

static const std::unordered_map<c10::ScalarType, int, ScalarTypeHashFunction> scalarTypeToONNXTypeMap = {
    {c10::kFloat, 1},
    {c10::kByte, 2},
    {c10::kChar, 3},
    {c10::kShort, 5},
    {c10::kInt, 6},
    {c10::kLong, 7},
    {c10::kBool, 9},
    {c10::kHalf, 10},
    {c10::kDouble, 11},
};

static int64_t ScalarTypeToONNXType(const c10::ScalarType& st) {
  int64_t onnx_type = -1;
  if (scalarTypeToONNXTypeMap.count(st) != 0) {
    onnx_type = scalarTypeToONNXTypeMap.find(st)->second;
  }
  return onnx_type;
}

static const std::unordered_set<NodeKind> arithmeticOps = {
  onnx::Add,
  onnx::Sub,
  onnx::Mul,
  onnx::Div,
  onnx::Gemm,
  onnx::Pow,
};

static bool IsArithmeticOp(const NodeKind& nkind) {
  return arithmeticOps.find(nkind) != arithmeticOps.end();
}

static const std::unordered_set<NodeKind> comparisonOps = {
  onnx::Greater,
  onnx::Less,
  onnx::Equal,
};

static bool IsComparisonOp(const NodeKind& nkind) {
  return comparisonOps.find(nkind) != comparisonOps.end();
}

static ProfiledTensorTypePtr CreateProfiledTensorTypeWithScalarType(
    const ProfiledTensorTypePtr& typePtr,
    const c10::ScalarType& scalar_type) {
  return ProfiledTensorType::create(
      scalar_type,
      typePtr->device(),
      typePtr->sizes(),
      typePtr->strides(),
      typePtr->requiresGrad());
}

static bool IsImplicitCastSupported(const NodeKind& nodeKind) {
  return (arithmeticOps.find(nodeKind) != arithmeticOps.end() ||
      comparisonOps.find(nodeKind) != comparisonOps.end());
}

static void ImplicitCastForONNXOnBlock(Block* block) {
  auto promoteScalarTypes = [&](const std::vector<c10::ScalarType>& types) -> c10::optional<c10::ScalarType> {
    if (types.empty()) {
      return c10::nullopt;
    }
    auto st = types[0];
    for (size_t i=1; i<types.size(); ++i) {
      st = c10::promoteTypes(st, types[i]);
    }
    return st;
  };

  auto inferExpectedScalarType = [&](const Node* n) -> c10::optional<c10::ScalarType> {
    std::vector<c10::ScalarType> typesFromTensors;
    std::vector<c10::ScalarType> typesFromScalars;
    std::for_each(n->inputs().begin(), n->inputs().end(), [&](const Value* input){
      auto nkind = input->node()->kind();
      if (nkind == onnx::Gather && input->node()->input(0)->node()->kind() == onnx::Shape) {
        typesFromScalars.emplace_back(c10::kLong);
      } else if (nkind == onnx::Constant) {
        typesFromScalars.emplace_back(input->node()->t(attr::value).scalar_type());
      } else if (auto scalar_type = ProfiledTensorType::create(input->type())->scalarType()) {
        typesFromTensors.emplace_back(*scalar_type);
      }
    });

    c10::optional<c10::ScalarType> st = c10::nullopt;
    const c10::optional<c10::ScalarType> output_st = ProfiledTensorType::create(n->output()->type())->scalarType();

    if (typesFromScalars.size() == n->inputs().size()) {
      // If all inputs are scalars, infer scalar_type by calling c10::promoteTypes.
      st = promoteScalarTypes(typesFromScalars);
    } else if (output_st) {
      st = output_st;
    } else if (!typesFromTensors.empty()) {
      st = typesFromTensors[0];
      if (std::any_of(typesFromTensors.begin(), typesFromTensors.end(), [&st](const c10::ScalarType& type) {
        return type != st;
      })) {
        std::cerr << "Warning: ONNX Scalar Type Analysis - Scalar types mismatch for tensor inputs of operator "
                  << n->kind().toDisplayString()
                  << ". Please report a bug to PyTorch. "
                  << "The scalar type of the first tensor is chosen." << std::endl;
      }
    } else {
      st = promoteScalarTypes(typesFromScalars);
    }

    return st;
  };

  auto updateScalarTypeForInputs = [&](Node* n, const c10::ScalarType& scalar_type) {
    const int64_t onnx_type = ScalarTypeToONNXType(scalar_type);
    if (onnx_type < 0) {
      return;
    }

    for (auto input : n->inputs()) {
      auto input_tensor_type = ProfiledTensorType::create(input->type());
      auto input_scalar_type = input_tensor_type->scalarType();

      if ((input->node()->kind() == onnx::Constant) ||
          (input_scalar_type && (*input_scalar_type != scalar_type))) {
        if (input->node()->kind() == onnx::Constant) {
          // Fix up the scalar directly instead of inserting a cast operator.
          // NOTE: Keep only the else branch once constant_folding is enabled by default.
          at::Tensor val = input->node()->t(attr::value);
          at::Tensor new_val = val.to(scalar_type);
          Node* const_node = block->owningGraph()->create(onnx::Constant);
          const_node->t_(attr::value, new_val);
          const_node->insertBefore(n);
          const_node->output()->setType(ProfiledTensorType::create(new_val));
          n->replaceInputWith(input, const_node->output());
        } else {
          Node* cast_node = block->owningGraph()->create(onnx::Cast);
          cast_node->addInput(input);
          cast_node->i_(attr::to, onnx_type);
          cast_node->insertBefore(n);
          cast_node->output()->setType(
              CreateProfiledTensorTypeWithScalarType(input_tensor_type, scalar_type));
          n->replaceInputWith(input, cast_node->output());
        }
      }
    }
  };

  auto updateScalarTypeForOutput = [&](Node* n, const c10::ScalarType& scalar_type) {
    auto output_tensor_type = ProfiledTensorType::create(n->output()->type());
    n->output()->setType(
        CreateProfiledTensorTypeWithScalarType(output_tensor_type, scalar_type));
  };

  for (auto it = block->nodes().begin(); it != block->nodes().end(); ++it) {
    for (auto sub : it->blocks()) {
      ImplicitCastForONNXOnBlock(sub);
    }
    WithInsertPoint guard(*it);
    auto* subgraph = it->owningGraph();

    if (IsImplicitCastSupported(it->kind())) {
      auto expected_scalar_type = inferExpectedScalarType(*it);
      if (expected_scalar_type) {
        updateScalarTypeForInputs(*it, *expected_scalar_type);
        if (!IsComparisonOp(it->kind())) {
          updateScalarTypeForOutput(*it, *expected_scalar_type);
        }
      }
    }
  }
  EliminateDeadCode(block, true, DCESideEffectPolicy::ALLOW_DELETING_NODES_WITH_SIDE_EFFECTS);
}

// This pass tries to resolve scalar type mismatch issues between input tensors
// introduced by the implicit type conversions on scalars.
void ImplicitCastForONNX(const std::shared_ptr<Graph>& graph) {
  ImplicitCastForONNXOnBlock(graph->block());
}
} // anonymous namespace


void ScalarTypeAnalysisForONNX(const std::shared_ptr<Graph>& graph) {
  ImplicitCastForONNX(graph);


}

} // namespace jit
} // namespace torch
