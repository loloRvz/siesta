#include <iostream>

#include "onnxruntime_cxx_api.h"

#define POSITION_HISTORY_LENGTH 8

int main(){
  // Declare stuff
  Ort::Env env;
  std::string model_path = "/home/lolo/omav_ws/src/rotors_simulator/rotors_description/models/T_a.pt";
  Ort::Session session(env, model_path, Ort::SessionOptions{ nullptr });
    


}