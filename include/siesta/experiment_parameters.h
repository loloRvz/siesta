#ifndef SIESTA_input_signal_params_h
#define SIESTA_input_signal_params_h

#include <string>

namespace experiment_parameters {

    // Default values
    int LOAD_ID = 1;

    double INPUT_FREQ = 200;
    enum INPUT_TYPES {STEP,CHRP,FLIT};
    int INPUT_TYPE = STEP; 
    double INPUT_CENTRE_POINT = 0;

    double STEP_FREQ = 5;
    double STEP_VARIANCE = 0.3;

    double CHRP_AMPLITUDE = 0.5;
    double CHRP_FREQ1 = 10;
    double CHRP_FREQ2 = 40;
    double CHRP_PERIOD = 3;

    double FLIT_FREQUENCY = 200;
    std::string FLIT_FILE = "23-02-08--21-26-11_ID5.csv";

    const std::string setpoint_topic_ = "set_position";

    /*** Load input signal parameters ***/
    void load_params(ros::NodeHandle nh){
        nh.getParam("/load_id",                   LOAD_ID);
        nh.getParam("/input/frequency",           INPUT_FREQ);
        nh.getParam("/input/type",                INPUT_TYPE);
        nh.getParam("/input/centre_point",        INPUT_CENTRE_POINT);

        nh.getParam("/input/gaussian/frequency",  STEP_FREQ);
        nh.getParam("/input/gaussian/variance",   STEP_VARIANCE);

        nh.getParam("/input/chirp/amplitude",     CHRP_AMPLITUDE);
        nh.getParam("/input/chirp/frequency1",    CHRP_FREQ1);
        nh.getParam("/input/chirp/frequency2",    CHRP_FREQ2);
        nh.getParam("/input/chirp/period",        CHRP_PERIOD);

        nh.getParam("/input/flight/rec_frequency",  FLIT_FREQUENCY);
        nh.getParam("/input/flight/file",           FLIT_FILE);
    }

}
#endif