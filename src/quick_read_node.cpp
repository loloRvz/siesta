#include <ros/ros.h>

#include "std_msgs/String.h"
#include "std_msgs/Int32.h"

#include "dynamixel_sdk/dynamixel_sdk.h"

using namespace dynamixel;

// Control table address
#define ADDR_TORQUE_ENABLE      64
#define ADDR_GOAL_POSITION      116
#define ADDR_REALTIME_TICK      120
#define ADDR_PRESENT_CURRENT    126
#define ADDR_PRESENT_VELOCITY   128
#define ADDR_PRESENT_POSITION   132

// Protocol version
#define PROTOCOL_VERSION        2.0 

// Default setting
#define DXL1_ID                 1
#define BAUDRATE                4000000
#define DEVICE_NAME             "/dev/ttyUSB0"


PortHandler * portHandler = PortHandler::getPortHandler(DEVICE_NAME);
PacketHandler * packetHandler = PacketHandler::getPacketHandler(PROTOCOL_VERSION);

GroupSyncRead groupSyncRead(portHandler, packetHandler, ADDR_GOAL_POSITION, 20);
GroupSyncWrite groupSyncWrite(portHandler, packetHandler, ADDR_GOAL_POSITION, 4);


void setupMotor() {
  uint8_t dxl_error = 0;
  int dxl_comm_result = COMM_TX_FAIL;

  if (!portHandler->openPort()) {
    ROS_ERROR("Failed to open the port!");
  }

  if (!portHandler->setBaudRate(BAUDRATE)) {
    ROS_ERROR("Failed to set the baudrate!");
  }

  dxl_comm_result = packetHandler->write1ByteTxRx(
    portHandler, DXL1_ID, ADDR_TORQUE_ENABLE, 1, &dxl_error);
  if (dxl_comm_result != COMM_SUCCESS) {
    ROS_ERROR("Failed to enable torque for Dynamixel ID %d", DXL1_ID);
  }
}

void setPositionCallback(std_msgs::Int32 msg) {
  uint32_t position = msg.data;

  uint8_t param_goal_position[4] = {DXL_LOBYTE(DXL_LOWORD(position)),
                                    DXL_HIBYTE(DXL_LOWORD(position)),
                                    DXL_LOBYTE(DXL_HIWORD(position)),
                                    DXL_HIBYTE(DXL_HIWORD(position))};

  bool dxl_addparam_result = groupSyncWrite.addParam(DXL1_ID, param_goal_position);
  int dxl_comm_result = groupSyncWrite.txPacket();
  if (dxl_comm_result == COMM_SUCCESS) {
    ROS_INFO("Wrote set_position: %d", position);
  } else {
    ROS_ERROR("Failed to set position! Error code: %d", dxl_comm_result);
  }
  groupSyncWrite.clearParam();
}

int main(int argc, char ** argv) {
  setupMotor();

  ros::init(argc, argv, "quick_read_node");
  ros::NodeHandle nh;
  ros::Subscriber set_position_sub = nh.subscribe("/set_position", 1000, setPositionCallback);
  ros::Rate rate(400);

  bool dxl_addparam_result = groupSyncRead.addParam(DXL1_ID);
  int dxl_comm_result = COMM_TX_FAIL;

  int32_t realtime_tick = 0;
  int32_t goal_position = 0;
  int32_t curr_position = 0;
  int32_t curr_velocity = 0;
  int32_t curr_current  = 0;

  while(ros::ok()){
    dxl_comm_result = groupSyncRead.txRxPacket();
    if (dxl_comm_result == COMM_SUCCESS) {
      realtime_tick = groupSyncRead.getData((uint8_t)DXL1_ID, ADDR_REALTIME_TICK, 2);
      goal_position = groupSyncRead.getData((uint8_t)DXL1_ID, ADDR_GOAL_POSITION, 4);
      curr_position = groupSyncRead.getData((uint8_t)DXL1_ID, ADDR_PRESENT_POSITION, 4);
      curr_velocity = groupSyncRead.getData((uint8_t)DXL1_ID, ADDR_PRESENT_VELOCITY, 4);
      curr_current = groupSyncRead.getData((uint8_t)DXL1_ID, ADDR_PRESENT_CURRENT, 2);

      ROS_INFO("Realtime tick %d", realtime_tick);
      ROS_INFO("Goal position %d", goal_position);
      ROS_INFO("Curr position %d", curr_position);
      ROS_INFO("Curr velocity %d", curr_velocity);
      ROS_INFO("Curr current %d", curr_current);
      ROS_INFO("********************");
    } else {
      ROS_ERROR("Connection failure! Error code: %d", dxl_comm_result);
    }

    ros::spinOnce();
    rate.sleep();
  }

  portHandler->closePort();
  return 0;
}
