#include <ros/ros.h>
#include <chrono>
#include <iostream>
#include <fstream>
#include <ctime>

#include "std_msgs/String.h"
#include "std_msgs/Float32.h"
#include "omav_hovery_interface/ll/dynamixel_motor_adapter.h"
#include "omav_hovery_interface/ll/polling_thread.h"

#include "motor_specs.h"

#define POLLING_FREQ 400

// Setpoint topic callback
class PositionSetter{
  public:
    void setPointCallback(std_msgs::Float32 msg) {
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

  // Init rosnode and subscribe to setpoint topic
  ros::init(argc, argv, "dxl_quick_read_node");
  ros::NodeHandle nh;
  ros::Subscriber set_position_sub = nh.subscribe("/set_position", 1000, &PositionSetter::setPointCallback, &ps);
  ros::Rate rate(POLLING_FREQ);

  // Declare motors & interfaces
  std::array<int, 1> dynamixels = {DXL1_ID};
  std::array<double, 1> setPointAngle{M_PI};
  omV::ll::MotorInterface<_POS>::MotorStatusArray readBackStatus;
  omV::ll::DynamixelMotorAdapter<_POS> ta_adapter(DEVICE_NAME, BAUDRATE, dynamixels);

  // Init motor
  ta_adapter.open();
  ta_adapter.enable();

  // Open file to write data
  time_t curr_time;
	tm * curr_tm;
	char filename_string[100], data_string[100];
  time(&curr_time);
	curr_tm = localtime(&curr_time);
  strftime(filename_string, 100, "/home/lolo/siesta_ws/src/siesta/data/%Y-%m-%d--%H-%M-%S_dataset.csv", curr_tm); // Create filename with date&time
  std::ofstream myfile;
  myfile.open(filename_string);
  myfile << "setpoint[rad],position[rad],velocity[rad/s],current[mA],delta_t[ms],acceleration[rad/s^2]\n"; // Set column descriptions

  // Time variables
  time_point t_now = std::chrono::system_clock::now();
  time_point t_prev = t_now;


  ROS_INFO("Polling motor...");
  while (ros::ok()) {
    // Measure exact loop frequency
    t_prev = t_now;
    t_now = std::chrono::system_clock::now();

    // Read motor data & write setpoint
    setPointAngle[0] = ps.getSetPoint();
    ta_adapter.write(setPointAngle);
    readBackStatus = ta_adapter.read();

    // Write data to csv file
    sprintf(data_string, "%03.2f,%03.2f,%03.2f,%03.2f,%03.2f,%03.2f\n",
            readBackStatus[0].setpoint, 
            readBackStatus[0].position, 
            readBackStatus[0].velocity,
            readBackStatus[0].current,  
            duration_cast<microseconds>(t_now - t_prev).count()/1000.0,
            NAN);
    myfile << data_string;

    // Loop
    ros::spinOnce();
    rate.sleep();
  }
  ROS_INFO("Stopped polling motor. Exiting...");

  myfile.close();
  return 0;
}
