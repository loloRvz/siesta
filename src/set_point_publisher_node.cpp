#include <ros/ros.h>
#include <random>
#include <sstream>

#include "std_msgs/String.h"
#include "std_msgs/Int32.h"

#include "motor_specs.h"


// Encoder data
#define POS_MIN                 0
#define POS_MAX                 4095

// Default setting
#define DXL1_ID                 1


int main(int argc, char ** argv) {

  ros::init(argc, argv, "set_point_publisher_node");
  ros::NodeHandle nh;
  ros::Publisher set_point_pub = nh.advertise<std_msgs::Int32>("/set_position", 1000);
  ros::Rate rate(10);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> distr(POS_MIN, POS_MAX);

  std_msgs::Int32 msg;

  while(ros::ok()){

    msg.data = distr(gen);

    ROS_INFO("Randomly generated set_position: %d", msg.data);

    set_point_pub.publish(msg);

    ros::spinOnce();
    rate.sleep();
  }

  return 0;
}
