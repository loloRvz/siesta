#include <ros/ros.h>
#include <random>
#include <sstream>
#include <fstream>
#include <vector> 

#include "std_msgs/String.h"
#include "std_msgs/Float32.h"

#include "experiment_parameters.h"

using namespace experiment_parameters;

/*** Input generator functions ***/
double rand_signal(double r){
	r = r<-M_PI ? -M_PI : r;              // Limit input range
	r = r>M_PI ? M_PI: r;
	return r;
}

double chirp_signal(double time){
	time = std::fmod(time,CHRP_PERIOD);
	return CHRP_AMPLITUDE*cos(CHRP_FREQ1*time+(CHRP_FREQ2-CHRP_FREQ1)*time*time/(2*CHRP_PERIOD)) + INPUT_CENTRE_POINT;
}

std::vector<std::vector<double>> parse_flight_data(std::string csv_path) {
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

double flight_signal(std::vector<std::vector<std::string>> data, int index){
	index %= data.size();
	return data[2][index];
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

	// RNG for random step input
	std::default_random_engine generator;
	std::normal_distribution<double> distribution(INPUT_CENTRE_POINT,sqrt(STEP_VARIANCE));
	int count = 0;

	// Open csv file for real data input
	std::vector<std::vector<double>> flight_data = parse_flight_data(std::string csv_path);
	int index = 0;

	ros::Rate rate(INPUT_FREQ);
	ROS_INFO("Publishing input...");
	while(ros::ok()){
		// Generate input depending on input type
		switch(INPUT_TYPE){

			case STEP:
				if(count > INPUT_FREQ/STEP_FREQ){
					msg.data = rand_signal(distribution(generator)); // Generate random variable with normal distribution
					count = 0;
				}else{count++;}
				break;

			case CHRP:
				msg.data = chirp_signal(ros::Time::now().toSec()); // Generate chirp signal
				break;

			case FLIT:
				msg.data = (flight_data,index); // Generate chirp signal
				index++;
				break;

			default:
				msg.data = -1;
		}

		// Publish set point
		set_point_pub.publish(msg);

		ros::spinOnce();
		rate.sleep();
	}

	ROS_INFO("Stopped publishing input. Exiting...");

	return 0;
}
