#include <ros/ros.h>
#include <random>
#include <sstream>

#include "std_msgs/String.h"
#include "std_msgs/Float32.h"

#include "motor_specs.h"

#define INPUT_VARIANCE 0.1


int main(int argc, char ** argv) {

  ros::init(argc, argv, "set_point_publisher_node");
  ros::NodeHandle nh;
  ros::Publisher set_point_pub = nh.advertise<std_msgs::Float32>("/set_position", 1000);
  ros::Rate rate(20);

  std_msgs::Float32 msg;

  std::default_random_engine generator;
  std::normal_distribution<double> distribution(M_PI,sqrt(INPUT_VARIANCE));
  double r;

  while(ros::ok()){
    r = distribution(generator);  // Generate random variable with normal distribution
    r = r<0 ? 0 : r;              // Limit input range
    r = r>2*M_PI ? 2*M_PI: r;  

    msg.data = r;
    ROS_INFO("Randomly generated set_position: %f", r);

    set_point_pub.publish(msg);

    ros::spinOnce();
    rate.sleep();
  }


  return 0;
}
