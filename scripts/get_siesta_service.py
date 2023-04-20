#!/usr/bin/env python

import rospy
import torch

from siesta.srv import GetSiesta



def handle_get_siesta(req):
    print(req)
    return 0.5


def get_siesta_server():
    rospy.init_node("get_siesta_server")
    s = rospy.Service("get_siesta", GetSiesta, handle_get_siesta)


if __name__ == '__main__':
    get_siesta_server()