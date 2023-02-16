#include <ros/ros.h>
#include <chrono>
#include <iostream>
#include <fstream>
#include <ctime>

#include "std_msgs/String.h"
#include "std_msgs/Float32.h"
#include "omav_hovery_interface/ll/dynamixel_motor_adapter.h"
#include "omav_hovery_interface/ll/polling_thread.h"

class PositionSetter{
  public:
    void setPositionCallback(std_msgs::Float32 msg) {
      set_point = msg.data;
    }
    double getSetPoint() {
      return set_point;
    }

  private:
    double set_point;
};


int main(int argc, char ** argv) {
  PositionSetter ps;

  // Init ros
  ros::init(argc, argv, "dxl_quick_read_node");
  ros::NodeHandle nh;
  ros::Subscriber set_position_sub = nh.subscribe("/set_position", 1000, &PositionSetter::setPositionCallback, &ps);
  ros::Rate rate(400);

  // Declare motors & interfaces
  std::array<int, 1> dynamixels = {1};
  std::array<double, 1> setPointAngle{M_PI};
  omV::ll::MotorInterface<_POS>::MotorStatusArray readBackStatus;
  omV::ll::DynamixelMotorAdapter<_POS> ta_adapter("/dev/ttyUSB0", 3000000, dynamixels);

  // Init motors
  ta_adapter.open();
  ta_adapter.enable();

  // Open file to write data
  time_t curr_time;
	tm * curr_tm;
	char filename_string[100], data_string[100];
  time(&curr_time);
	curr_tm = localtime(&curr_time);
  strftime(filename_string, 100, "/home/lolo/siesta_ws/src/siesta/data/%Y-%m-%d--%H-%M-%S_dataset.csv", curr_tm);

  std::ofstream myfile;
  myfile.open(filename_string);
  myfile << "setpoint [rad], position [rad], velocity [rad/s], current [mA], delta_t [ms]\n";

  // Wait for keystroke to start
  std::getchar();

  // Loop frequency measurements
  time_point t_now = std::chrono::system_clock::now();
  time_point t_prev = t_now;


  while (ros::ok()) {
    t_prev = t_now;
    t_now = std::chrono::system_clock::now();

    setPointAngle[0] = ps.getSetPoint();
    ta_adapter.write(setPointAngle);
    readBackStatus = ta_adapter.read();

    sprintf(data_string, "%03.2f, %03.2f, %03.2f, %03.2f, %03.2f\n",
            readBackStatus[0].setpoint, 
            readBackStatus[0].position, 
            readBackStatus[0].velocity,
            readBackStatus[0].current,  
            duration_cast<microseconds>(t_now - t_prev).count()/1000.0);
    myfile << data_string;

    printf("setpt [rad]: %03.2f \t pos [rad]: %03.2f \t vel [rad/s]: %03.2f \t amp [mA]: %03.2f \t delta_t [ms]= %03.2f\n",
           readBackStatus[0].setpoint, 
           readBackStatus[0].position, 
           readBackStatus[0].velocity,
           readBackStatus[0].current,  
           duration_cast<microseconds>(t_now - t_prev).count()/1000.0);

    ros::spinOnce();
    rate.sleep();
  }

  myfile.close();
  return 0;
}
