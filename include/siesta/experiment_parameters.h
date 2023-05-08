#ifndef SIESTA_input_signal_params_h
#define SIESTA_input_signal_params_h

#include <string>

namespace experiment_parameters {

    // Default values
    int LOAD_ID = 1;

    int CTRL_FREQ = 200;

    int SMPL_FREQ = 400;
    
    enum INPUT_TYPES {STEP,CHRP,FLIT,MIXD,NOIS, N_INPUT_TYPES};
    int INPUT_TYPE = STEP; 

    const std::string setpoint_topic_ = "set_position";

    /*** Load input signal parameters ***/
    void load_params(ros::NodeHandle nh){
        nh.getParam("load_id",             LOAD_ID);
        nh.getParam("control_frequency",   CTRL_FREQ);
        nh.getParam("sample_frequency",    SMPL_FREQ);
        nh.getParam("input_type",          INPUT_TYPE);
    }

}
#endif