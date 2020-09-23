#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

TORCH_API TypePtr
MergeInferredType(TypePtr existing_type, TypePtr inferred_type);

TORCH_API std::shared_ptr<Graph> ONNXSetDynamicInputShape(
    std::shared_ptr<Graph>& graph,
    const std::unordered_map<
        std::string,
        std::unordered_map<int64_t, std::string>>& dynamic_axes,
    const std::vector<std::string>& input_names);

TORCH_API std::shared_ptr<Graph> ONNXAssignOutputShape(
    std::shared_ptr<Graph>& graph,
    at::ArrayRef<at::Tensor> outputs,
    bool onnx_shape_inference);

// Utilize ONNX Shape Inference for node.
// The node must have ONNX namespace, and is valid ONNX node accroding to spec.
// On successful ONNX shape inference runs, the function updates output types of
// n with inferred shape and type. Otherwise n is unchanged.
TORCH_API void ONNXShapeTypeInference(Node* n, int opset_version);

TORCH_API std::shared_ptr<Graph> ONNXShapeTypeInference(std::shared_ptr<Graph>& g, int opset_version);

} // namespace jit
} // namespace torch
