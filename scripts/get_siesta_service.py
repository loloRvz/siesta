#!/usr/bin/env python
import rospy
import torch

from siesta.srv import GetSiesta

PHL = 8

class TorqueEstimator:
    def __init__(self):
        # Load torch model
        self.model_dir = "/root/catkin_ws/src/siesta/data/models/saved/23-03-29--10-08-46_400Hz-L9-mixd-PHL08_Ta/delta_2000.pt"
        self.model = torch.jit.load(self.model_dir)

        # Init rossrv
        self.siesta_serv = rospy.Service("get_siesta", GetSiesta, self.get_siesta_cb)

    def get_siesta_cb(self, request):
        if(len(request.peh) != PHL):
            print("Wrong input size for PHL: ", PHL)
            return 0
        new_peh = request.peh[::-1]
        torque = self.model(torch.tensor(new_peh))
        #rospy.loginfo("Torque estimation: %f",torque)
        return torque


if __name__ == '__main__':
    rospy.init_node("get_siesta_server")
    siesta = TorqueEstimator()
    rospy.spin()