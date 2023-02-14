#!/usr/bin/env python

import os
import rospy
from dynamixel_sdk import *
from dynamixel_sdk_examples.srv import *
from dynamixel_sdk_examples.msg import *

if os.name == 'nt':
    import msvcrt
    def getch():
        return msvcrt.getch().decode()
else:
    import sys, tty, termios
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    def getch():
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

# Control table address
ADDR_TORQUE_ENABLE      = 64               # Control table address is different in Dynamixel model
ADDR_GOAL_POSITION      = 116
ADDR_REALTIME_TICK      = 120
ADDR_PRESENT_CURRENT    = 126
ADDR_PRESENT_VELOCITY   = 128
ADDR_PRESENT_POSITION   = 132

# Protocol version
PROTOCOL_VERSION            = 2.0               # See which protocol version is used in the Dynamixel

# Default setting
DXL_ID                      = 1                 # Dynamixel ID : 1
BAUDRATE                    = 4000000           # Dynamixel default baudrate : 57600
DEVICENAME                  = '/dev/ttyUSB0'    # Check which port is being used on your controller
                                                # ex) Windows: "COM1"   Linux: "/dev/ttyUSB0" Mac: "/dev/tty.usbserial-*"

TORQUE_ENABLE               = 1                 # Value for enabling the torque
TORQUE_DISABLE              = 0                 # Value for disabling the torque
DXL_MINIMUM_POSITION_VALUE  = 0               # Dynamixel will rotate between this value
DXL_MAXIMUM_POSITION_VALUE  = 1000            # and this value (note that the Dynamixel would not move when the position value is out of movable range. Check e-manual about the range of the Dynamixel you use.)
DXL_MOVING_STATUS_THRESHOLD = 20                # Dynamixel moving status threshold

portHandler = PortHandler(DEVICENAME)
packetHandler = PacketHandler(PROTOCOL_VERSION)

def setup_motor():
    # Open port
    try:
       portHandler.openPort()
       print("Succeeded to open the port")
    except:
        print("Failed to open the port")
        print("Press any key to terminate...")
        getch()
        quit()
    # Set port baudrate
    try:
        portHandler.setBaudRate(BAUDRATE)
        print("Succeeded to change the baudrate")
    except:
        print("Failed to change the baudrate")
        print("Press any key to terminate...")
        getch()
        quit()

    # Enable Dynamixel Torque
    dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)
    if dxl_comm_result != COMM_SUCCESS:
        print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
        print("Press any key to terminate...")
        getch()
        quit()
    elif dxl_error != 0:
        print("%s" % packetHandler.getRxPacketError(dxl_error))
        print("Press any key to terminate...")
        getch()
        quit()
    else:
        print("DYNAMIXEL has been successfully connected")

    print("Ready to get & set Position.")

def set_goal_pos_callback(data):
    print("Set Goal Position of ID %s = %s" % (data.id, data.position))
    dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(portHandler, data.id, ADDR_GOAL_POSITION, data.position)

def main():
    print("Setting up motor...")
    setup_motor()

    rospy.init_node('read_write_py_node')
    rospy.Subscriber('set_position', SetPosition, set_goal_pos_callback)


    while not rospy.is_shutdown():
        dxl_realtime_tick, dxl_comm_result, dxl_error = packetHandler.read4ByteTxRx(portHandler, DXL_ID, ADDR_REALTIME_TICK)
        dxl_goal_position, dxl_comm_result, dxl_error = packetHandler.read4ByteTxRx(portHandler, DXL_ID, ADDR_PRESENT_POSITION)
        dxl_present_position, dxl_comm_result, dxl_error = packetHandler.read4ByteTxRx(portHandler, DXL_ID, ADDR_PRESENT_POSITION)
        dxl_present_velocity, dxl_comm_result, dxl_error = packetHandler.read4ByteTxRx(portHandler, DXL_ID, ADDR_PRESENT_VELOCITY)
        dxl_present_current, dxl_comm_result, dxl_error = packetHandler.read4ByteTxRx(portHandler, DXL_ID, ADDR_PRESENT_CURRENT)
        
        print("Realtime tick = ", dxl_realtime_tick)
        print("Goal Position = ", dxl_goal_position)
        print("Curr Position = ", dxl_present_position)
        print("Curr Velocity = ", dxl_present_velocity)
        print("Curr Current  = ", dxl_present_current)
        print("********************")
        

if __name__ == '__main__':
    main()
