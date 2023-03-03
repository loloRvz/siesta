#! /usr/bin/env python3

import rosbag
import os, glob
import pandas as pd
import csv

from omav_hovery_msgs.msg import UAVStatus

### SCRIPT ###
def main():
    # Get bag file name
    dir_path = os.path.dirname(os.path.realpath(__file__))
    list_of_files = glob.glob(dir_path + '/../bags/*.bag')
    bag_path = max(list_of_files, key=os.path.getctime)

    # Open bag file
    bag = rosbag.Bag(bag_path)
    start_time = bag.get_start_time()
    motor_id = 6

    for motor_id in range(6,12):
        csv_path = dir_path + '/../data_flight/' + os.path.basename(bag_path)[2:10] + "--" + os.path.basename(bag_path)[11:-4] + "_ID" + str(motor_id-6) + ".csv"
        with open(csv_path, 'w', newline='') as file:
            writer = csv.writer(file)
            # Write header row
            writer.writerow(["time[ms]","setpoint[rad]","position[rad]","velocity[rad/s]","current[mA]","velocity_computed[rad/s]","acceleration_computed[rad/s^2]"])
            for topic, msg, t in bag.read_messages(topics=['/stork/uav_state']):
                # Write one msg info to csv
                writer.writerow([( \
                    msg.header.stamp.to_sec()-start_time)*1000, \
                    msg.motors[motor_id].setpoint, \
                    msg.motors[motor_id].position, \
                    'nan', \
                    'nan', \
                    'nan', \
                    'nan'])

    # Close bag
    bag.close()




if __name__ == "__main__":
    main()

