cmake_minimum_required(VERSION 3.0.2)
project(siesta)

add_definitions(-std=c++17 -fPIC -Wall -Wno-reorder -Wno-unknown-pragmas)
add_compile_definitions(MODE_DEFAULT)

find_package(PythonLibs REQUIRED)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  roslib
  omav_hovery_interface
  message_generation
)

## Generate messages in the 'msg' folder
# add_message_files(
#   FILES
#   Message1.msg
#   Message2.msg
# )

# Generate services in the 'srv' folder
add_service_files(FILES
  GetSiesta.srv
)

## Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

catkin_python_setup()

# Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_install_python(PROGRAMS
  setup.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

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

add_executable(test_libtorch src/test_libtorch.cpp)
add_dependencies(test_libtorch ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(test_libtorch ${catkin_LIBRARIES})

add_executable(test_onnx src/test_onnx.cpp)
add_dependencies(test_onnx ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(test_onnx ${catkin_LIBRARIES})



set(CMAKE_PREFIX_PATH "~/omav_ws/src/libtorch:~/omav_ws/src/onnxruntime:/opt/ros/noetic")

#include_directories(${ADDITIONAL_INCLUDE_DIRS} "/usr/include/onnxruntime")
include_directories(${ADDITIONAL_INCLUDE_DIRS} "/home/lolo/omav_ws/src/onnxruntime/include")

#/home/lolo/.local/share/cmake


find_package(Torch REQUIRED)
if(Torch_FOUND)
  message(WARNING "Torch FOUND, building torch modules.")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
else()
  message(WARNING "Torch NOT FOUND, skipping torch modules.")
endif()

if(Torch_FOUND)
target_link_libraries(test_libtorch "${TORCH_LIBRARIES}") 
SET_TARGET_PROPERTIES(test_libtorch PROPERTIES CXX_STANDARD 14)
endif()


find_package(onnxruntime REQUIRED)
if(onnxruntime_FOUND)
  message(WARNING "Onnx FOUND, building onnx modules.")
else()
  message(WARNING "Torch NOT FOUND, skipping torch modules.")
endif()

if(onnxruntime_FOUND)
target_link_libraries(test_onnx "${onnxruntime_LIBRARIES}") 
SET_TARGET_PROPERTIES(test_onnx PROPERTIES CXX_STANDARD 14)
endif()