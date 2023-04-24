#include <torch/torch.h>
#include <torch/script.h>

#define POSITION_HISTORY_LENGTH 8

int main(){
    // Declare stuff
    torch::jit::script::Module policy_;
    float pos_err_hist_ [POSITION_HISTORY_LENGTH] = {0,0,0,0,0,0,0.5,-1};
    std::vector<torch::jit::IValue> input;
    torch::Tensor model_input;
    torch::Tensor model_output;

    // Init model and position error history array
    try {
      policy_ = torch::jit::load("/root/catkin_ws/src/rotors_simulator/rotors_description/models/T_a.pt");
    } catch (const c10::Error& e){
      std::cerr << " Error loading the model\n";
    }
    std::cout << "model loaded ok\n";

    // Prepare inputs
    model_input = torch::zeros({1, POSITION_HISTORY_LENGTH});
    for(int i = 0; i < POSITION_HISTORY_LENGTH; i++){
        model_input[0].index_put_({i}, pos_err_hist_[i]);
    }
    input.push_back(model_input);

    // Compute output
    try {
      model_output = policy_.forward(input).toTensor();
    } catch (const c10::Error& e){
      std::cerr << " Error forward pass\n";
    }

    std::cout << "Input: " << model_input << std::endl;
    std::cout << "Output: " << model_output << std::endl;
    


}