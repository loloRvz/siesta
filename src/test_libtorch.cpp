#include <torch/torch.h>
#include <torch/script.h>

#define POSITION_HISTORY_LENGTH 8

int main(){
    // Declare stuff
    std::vector<float> pos_err_hist_ = {0,0,0,0,0,0,0.5,-1};

    torch::jit::script::Module policy_;
    at::Tensor input_tensor;
    at::Tensor output_tensor;
    std::vector<torch::jit::IValue> input_vect;
    at::TensorOptions options = torch::TensorOptions().dtype(torch::kFloat32);

    // Init model and position error history array
    try {
      policy_ = torch::jit::load("/home/lolo/omav_ws/src/rotors_simulator/rotors_description/models/T_a.pt");
    } catch (const c10::Error& e){
      std::cerr << " Error loading the model\n";
    }
    std::cout << "model loaded ok\n";
    policy_.eval();
 
    input_tensor = torch::from_blob(pos_err_hist_.data(), {1,POSITION_HISTORY_LENGTH}, options);
    input_vect.push_back(input_tensor);

    // Compute output
    try {
      output_tensor = policy_.forward(input_vect).toTensor();
    } catch (const c10::Error& e){
      std::cerr << " Error forward pass\n";
    }

    std::cout << "Input: " << input_vect << std::endl;
    std::cout << input_vect.size() << std::endl;
    std::cout << "Output: " << output_tensor << std::endl;
    


}