#ifndef SIESTA_input_signal_params_h
#define SIESTA_input_signal_params_h

#include <string>

namespace input_signal_params {

    // Default values
    double INPUT_FREQ = 200;
    enum INPUT_TYPES {GAUS,CHRP};
    int INPUT_TYPE = GAUS; 
    double INPUT_CENTRE_POINT = 3.1415926536;

    double GAUS_FREQ = 5;
    double GAUS_VARIANCE = 0.3;

    double CHRP_AMPLITUDE = 0.5;
    double CHRP_FREQ1 = 10;
    double CHRP_FREQ2 = 40;
    double CHRP_PERIOD = 3;

}
#endif