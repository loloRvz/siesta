#include <chrono>
#include <iostream>

#include "omav_hovery_interface/ll/dynamixel_motor_adapter.h"
#include "omav_hovery_interface/ll/polling_thread.h"

using namespace std::chrono;

int main() {
  std::array<int, 1> dynamixels = {1};
  std::array<double, 1> setPointAngle{M_PI};

  omV::ll::MotorInterface<_POS>::MotorStatusArray readBackStatus;
  omV::ll::DynamixelMotorAdapter<_POS> ta_adapter("/dev/ttyUSB0", 3000000, dynamixels);

  ta_adapter.open();
  ta_adapter.disable();
  ta_adapter.enable();

  std::getchar();

  while (1) {
    microseconds t1 = duration_cast<microseconds>(system_clock::now().time_since_epoch());
    ta_adapter.write(setPointAngle);

    readBackStatus = ta_adapter.read();
    microseconds t2 = duration_cast<microseconds>(system_clock::now().time_since_epoch());

    printf("%03.2f \t %03.2f \t %03.2f \t %03.2f \t %03.2f  \t delta_t= %03.2f ms \n",
           readBackStatus[0].position, readBackStatus[1].position, readBackStatus[2].position,
           readBackStatus[3].position, readBackStatus[4].position, (t2 - t1).count() / 1000.0);
  }
#pragma clang diagnostic pop
  return 0;
}
