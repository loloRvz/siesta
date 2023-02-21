#include <ros/ros.h>
#include <random>
#include <sstream>

#include "std_msgs/String.h"
#include "std_msgs/Float32.h"

#include "input_signal_params.h"

using namespace input_signal_params;

void load_params(ros::NodeHandle nh){
  nh.getParam("/input/frequency",           INPUT_FREQ);
  nh.getParam("/input/type",                INPUT_TYPE);
  nh.getParam("/input/centre_point",        INPUT_CENTRE_POINT);

  nh.getParam("/input/gaussian/frequency",  GAUS_FREQ);
  nh.getParam("/input/gaussian/variance",  GAUS_VARIANCE);

  nh.getParam("/input/chirp/amplitude",     CHRP_AMPLITUDE);
  nh.getParam("/input/chirp/frequency1",    CHRP_FREQ1);
  nh.getParam("/input/chirp/frequency2",    CHRP_FREQ2);
  nh.getParam("/input/chirp/period",        CHRP_PERIOD);
}

/*** Input generator functions ***/
double rand_signal(double r){
  r = r<0 ? 0 : r;              // Limit input range
  r = r>2*M_PI ? 2*M_PI: r;
  return r;
}

double chirp_signal(double time){
  time = std::fmod(time,CHRP_PERIOD);
  return CHRP_AMPLITUDE*cos(CHRP_FREQ1*time+(CHRP_FREQ2-CHRP_FREQ1)*time*time/(2*CHRP_PERIOD)) + INPUT_CENTRE_POINT;
}


/*** MAIN ***/
int main(int argc, char ** argv) {
  //Init ros stuff
  ros::init(argc, argv, "set_point_publisher_node");
  ros::NodeHandle nh;
  ros::Publisher set_point_pub = nh.advertise<std_msgs::Float32>("/set_position", 1000);
  std_msgs::Float32 msg;

  //Parameters
  load_params(nh);

  //RNG
  std::default_random_engine generator;
  std::normal_distribution<double> distribution(INPUT_CENTRE_POINT,sqrt(GAUS_VARIANCE));
  int count = 0;

  ros::Rate rate(INPUT_FREQ);
  ROS_INFO("Publishing input...");
  while(ros::ok()){
    // Generate input depending on input type
    switch(INPUT_TYPE){

      case GAUS:
        if(count > INPUT_FREQ/GAUS_FREQ){
          msg.data = rand_signal(distribution(generator)); // Generate random variable with normal distribution
          count = 0;
        }else{count++;}
        break;

      case CHRP:
        msg.data = chirp_signal(ros::Time::now().toSec()); // Generate chirp signal
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
