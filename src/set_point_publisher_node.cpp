#include <ros/ros.h>
#include <random>
#include <sstream>
#include <fstream>
#include <vector> 

#include "std_msgs/String.h"
#include "std_msgs/Float32.h"

#include "experiment_parameters.h"

using namespace experiment_parameters;


// Parse CSV function
std::vector<std::vector<double>> parse_csv(std::string csv_path) {
	std::ifstream  data(csv_path);
	std::string line;
	std::vector<std::vector<double>> parsedCsv;
	while(std::getline(data,line)){
		std::stringstream lineStream(line);
		std::string cell;
		std::vector<double> parsedRow;
		while(std::getline(lineStream,cell,',')){
			parsedRow.push_back(std::stod(cell));
		}
		parsedCsv.push_back(parsedRow);
	}
	return parsedCsv;  
}



/*** MAIN ***/
int main(int argc, char ** argv) {
	//Init rosnode and setpoint publisher
	ros::init(argc, argv, "set_point_publisher_node");
	ros::NodeHandle nh;
	ros::Publisher set_point_pub = nh.advertise<std_msgs::Float32>("/set_position", 1000);
	std_msgs::Float32 msg;

	// Get experiment parameters
	load_params(nh);

	// Get input signals from csv files
	std::vector<std::vector<double>> input_signals = parse_csv("../data/input_signals/signals.csv");
	int input_idx = 0;

	// Check for incorrect input type
	if(INPUT_TYPE > N_INPUT_TYPES){
		ROS_ERROR("Input type not valid. Exiting...");
		return 0;
	}

	ros::Rate rate(CTRL_FREQ);
	ROS_INFO("Publishing input...");
	while(ros::ok()){

		// Get input value
		msg.data = input_signals[INPUT_TYPE][input_idx];
		input_idx++;
		if (input_idx >= (int) input_signals[0].size()) {
			input_idx = 0;
		}

		// Publish set point
		set_point_pub.publish(msg);

		ros::spinOnce();
		rate.sleep();
	}

	ROS_INFO("Stopped publishing input. Exiting...");

	return 0;
}
