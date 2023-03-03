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
#include "experiment_parameters.h"

using namespace experiment_parameters;

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
		double set_point = 0;
};


int main(int argc, char ** argv) {

	// Init rosnode and subscribe to setpoint topic with callback to position setter
	ros::init(argc, argv, "dxl_quick_read_node");
	ros::NodeHandle nh;
	PositionSetter ps;
	ros::Subscriber set_position_sub = nh.subscribe(setpoint_topic_, 1000, &PositionSetter::setPointCallback, &ps);
	ros::Rate rate(POLLING_FREQ);

	// Get experiment parameters
	load_params(nh);

	// Declare motors & interfaces
	std::array<int, 1> dynamixels = {DXL1_ID};
	std::array<double, 1> setPointAngle{M_PI};
	omV::ll::MotorInterface<_POS>::MotorStatusArray readBackStatus;
	omV::ll::DynamixelMotorAdapter<_POS> ta_adapter(DEVICE_NAME, BAUDRATE, dynamixels);

	// Init motor
	ta_adapter.open();
	ta_adapter.enable();

	// Create filename for exprmt data
	time_t curr_time; 
	tm * curr_tm;
	char file_str[100], time_str[100], exprmt_descr_str[100], data_str[100];
	strcpy(file_str, "/home/lolo/siesta_ws/src/siesta/data/"); //Global path
	time(&curr_time);
	curr_tm = localtime(&curr_time);
	strftime(time_str, 100, "%y-%m-%d--%H-%M-%S_", curr_tm);
	strcat(file_str,time_str);  // Add date & time
	sprintf(exprmt_descr_str, "L%d",LOAD_ID);
	strcat(file_str,exprmt_descr_str); //Add load id
	//Add input type
	if(INPUT_TYPE == STEP){
		strcat(file_str,"-step");
	}else if(INPUT_TYPE == CHRP){
		strcat(file_str,"-chrp");
	}else if(INPUT_TYPE == FLIT){
		strcat(file_str,"-flit");
	}
	strcat(file_str,".csv");

	// Open file to write data
	std::ofstream myfile;
	myfile.open(file_str);
	myfile << "time[ms],setpoint[rad],position[rad],velocity[rad/s],current[mA],velocity_computed[rad/s],acceleration_computed[rad/s^2]\n"; // Set column descriptions

	// Time variables
	time_point t_now = std::chrono::system_clock::now();
	time_point t_start = std::chrono::system_clock::now();

	// Wait for first setpoint topic to be published
	ros::topic::waitForMessage<std_msgs::Float32>(setpoint_topic_,ros::Duration(5));

	ROS_INFO("Polling motor...");
	while (ros::ok()) {
		// Measure exact loop time
		t_now = std::chrono::system_clock::now();

		// Read motor data & write setpoint
		setPointAngle[0] = ps.getSetPoint() + M_PI;
		ta_adapter.write(setPointAngle);
		readBackStatus = ta_adapter.read();

		// Write data to csv file
		sprintf(data_str, "%010.3f,%07.5f,%07.5f,%06.3f,%08.2f,%03.3f,%03.3f\n",
			duration_cast<microseconds>(t_now - t_start).count()/1000.0,
			readBackStatus[0].setpoint - M_PI, 
			readBackStatus[0].position - M_PI, 
			readBackStatus[0].velocity,
			readBackStatus[0].current,
			NAN,
			NAN);
		myfile << data_str;

		// Loop
		ros::spinOnce();
		rate.sleep();
	}
	ROS_INFO("Stopped polling motor. Exiting...");

	ta_adapter.disable();
	myfile.close();
	return 0;
}
