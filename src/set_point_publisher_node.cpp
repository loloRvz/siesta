#include <ros/ros.h>
#include <random>
#include <sstream>

#include "std_msgs/String.h"
#include "std_msgs/Float32.h"

#include "motor_specs.h"

#define INPUT_FREQ 200
enum INPUT_TYPES {RAND,CHRP};
#define INPUT_TYPE RAND // 0: normal distr. ; 1: chirp signal 
#define INPUT_MEAN M_PI

#define RAND_SWITCH_FREQ 10
#define RAND_VARIANCE 0.5

#define CHRP_AMPLITUDE 0.5
#define CHRP_FREQ1 2
#define CHRP_FREQ2 30
#define CHRP_AMPLITUDE 0.5
#define CHRP_PERIOD 8

/*** Input generator functions ***/
double rand_signal(double r){
  r = r<0 ? 0 : r;              // Limit input range
  r = r>2*M_PI ? 2*M_PI: r;
  return r;
}

double chirp_signal(double w1, double w2, double A, double T, double time){
  time = std::fmod(time,T);
  return A*cos(w1*time+(w2-w1)*time*time/(2*T)) + INPUT_MEAN;
}


/*** MAIN ***/
int main(int argc, char ** argv) {
  //Init ros stuff
  ros::init(argc, argv, "set_point_publisher_node");
  ros::NodeHandle nh;
  ros::Publisher set_point_pub = nh.advertise<std_msgs::Float32>("/set_position", 1000);
  ros::Rate rate(INPUT_FREQ);
  std_msgs::Float32 msg;

  //RNG
  std::default_random_engine generator;
  std::normal_distribution<double> distribution(INPUT_MEAN,sqrt(RAND_VARIANCE));
  int count = 0;

  ROS_INFO("Publishing input...");
  while(ros::ok()){
    switch(INPUT_TYPE){
      case RAND:
        if(count > INPUT_FREQ/RAND_SWITCH_FREQ){
          msg.data = rand_signal(distribution(generator)); // Generate random variable with normal distribution
          count = 0;
        }else{
          count++;
        }
        break;

      case CHRP:
        msg.data = chirp_signal(CHRP_FREQ1, CHRP_FREQ2, CHRP_AMPLITUDE, CHRP_PERIOD, ros::Time::now().toSec()); // Generate chirp signal
        break;
        
      default:
        msg.data = -1;
    }
  
    set_point_pub.publish(msg);

    ros::spinOnce();
    rate.sleep();
  }

  ROS_INFO("Stopped publishing input. Exiting...");

  return 0;
}
