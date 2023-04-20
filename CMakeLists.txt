cmake_minimum_required(VERSION 3.0.2)
project(siesta)

add_definitions(-std=c++17 -fPIC -Wall -Wno-reorder -Wno-unknown-pragmas)
add_compile_definitions(MODE_DEFAULT)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  roslib
  omav_hovery_interface
  message_generation
  #dynamixel_sdk
  #dynamixel_sdk_examples
)

## Generate messages in the 'msg' folder
# add_message_files(
#   FILES
#   Message1.msg
#   Message2.msg
# )

# Generate services in the 'srv' folder
add_service_files(
  FILES
  GetSiesta.srv
)

## Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

## Generate added messages and services with any dependencies listed here
# generate_messages(
#   DEPENDENCIES
#   std_msgs
# )

## Generate dynamic reconfigure parameters in the 'cfg' folder
# generate_dynamic_reconfigure_options(
#   cfg/DynReconf1.cfg
#   cfg/DynReconf2.cfg
# )

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS roscpp rospy
)

include_directories(
  include/siesta
  ${catkin_INCLUDE_DIRS}
  ${YAML_INCLUDE_DIRS}
)


add_executable(dxl_quick_read_node src/dxl_quick_read_node.cpp)
add_dependencies(dxl_quick_read_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(dxl_quick_read_node ${catkin_LIBRARIES})

add_executable(set_point_publisher_node src/set_point_publisher_node.cpp)
add_dependencies(set_point_publisher_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(set_point_publisher_node ${catkin_LIBRARIES})
