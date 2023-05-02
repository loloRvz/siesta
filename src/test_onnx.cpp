#include <iostream>
#include <assert.h>

#include <onnxruntime_cxx_api.h>

#define POSITION_HISTORY_LENGTH 8

int main(){
  std::vector<float> pos_err_hist_ = {0,0,0,0,0,0,0.5,-1};

  // Load model
  Ort::Env env;
  const char* model_path = "/home/lolo/omav_ws/src/rotors_simulator/rotors_description/models/T_a.onnx";
  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(1);
  session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
  Ort::Session session(env, model_path, session_options);

  Ort::AllocatorWithDefaultOptions allocator;

  // Input nodes
  const size_t num_input_nodes = session.GetInputCount();
  std::vector<Ort::AllocatedStringPtr> input_names_ptr;
  std::vector<const char*> input_node_names;
  input_names_ptr.reserve(num_input_nodes);
  input_node_names.reserve(num_input_nodes);
  std::vector<int64_t> input_node_dims;
  std::cout << "Number of inputs = " << num_input_nodes << std::endl;
  for (size_t i = 0; i < num_input_nodes; i++) {
    // print input node names
    auto input_name = session.GetInputNameAllocated(i, allocator);
    std::cout << "Input " << i << " : name =" << input_name.get() << std::endl;
    input_node_names.push_back(input_name.get());
    input_names_ptr.push_back(std::move(input_name));

     // print input node types
    auto type_info = session.GetInputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType type = tensor_info.GetElementType();
    std::cout << "Input " << i << " : type = " << type << std::endl;

    // print input shapes/dims
    input_node_dims = tensor_info.GetShape();
    std::cout << "Input " << i << " : num_dims = " << input_node_dims.size() << '\n';
    for (size_t j = 0; j < input_node_dims.size(); j++) {
      std::cout << "Input " << i << " : dim[" << j << "] =" << input_node_dims[j] << '\n';
    }
    std::cout << std::flush;
  }
  // Init input vector
  constexpr size_t input_tensor_size = 8;
  std::vector<float> input_tensor_values(input_tensor_size);
  for (unsigned int i = 0; i < input_tensor_size; i++) {
    input_tensor_values[i] = pos_err_hist_[i];
  }
  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  auto input_tensor = Ort::Value::CreateTensor<float>(memory_info, 
                                                      input_tensor_values.data(), 
                                                      input_tensor_size, 
                                                      input_node_dims.data(), 
                                                      1);
  assert(input_tensor.IsTensor());

  // Output nodes
  const size_t num_output_nodes = session.GetOutputCount();
  std::vector<Ort::AllocatedStringPtr> output_names_ptr;
  std::vector<const char*> output_node_names;
  output_names_ptr.reserve(num_output_nodes);
  output_node_names.reserve(num_output_nodes);
  std::vector<int64_t> output_node_dims;
  std::cout << "Number of outputs = " << num_output_nodes << std::endl;
  for (size_t i = 0; i < num_output_nodes; i++) {
    // print output node names
    auto output_name = session.GetOutputNameAllocated(i, allocator);
    std::cout << "Outnput " << i << " : name =" << output_name.get() << std::endl;
    output_node_names.push_back(output_name.get());
    output_names_ptr.push_back(std::move(output_name));

     // print input node types
    auto type_info = session.GetOutputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType type = tensor_info.GetElementType();
    std::cout << "Output " << i << " : type = " << type << std::endl;

    // print input shapes/dims
    output_node_dims = tensor_info.GetShape();
    std::cout << "Output " << i << " : num_dims = " << output_node_dims.size() << '\n';
    for (size_t j = 0; j < input_node_dims.size(); j++) {
      std::cout << "Output " << i << " : dim[" << j << "] =" << output_node_dims[j] << '\n';
    }
    std::cout << std::flush;
  }
  // Output tensor
  auto output_tensors = session.Run(Ort::RunOptions{nullptr}, 
                                    input_node_names.data(), 
                                    &input_tensor, 
                                    1, 
                                    output_node_names.data(), 1);
  assert(output_tensors.size() == 1 && output_tensors.front().IsTensor());
    
  // Get pointer to output tensor float values
  float* floatarr = output_tensors.front().GetTensorMutableData<float>();
  std::cout << "Output: " << floatarr[0] << "\n";


}