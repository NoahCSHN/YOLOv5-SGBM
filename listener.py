#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Description:       :
@Date     :2021/04/23 21:50:37
@Author      :xia
@version      :1.0
'''
import rospy,cv2,socket,numpy,time
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import PolygonStamped
from multiprocessing import Process

def Axiscallback(data):
    rospy.loginfo(rospy.get_caller_id() + 'I heard %s', data)

def Imgcallback(data):
#    print('1')
    bridge = CvBridge()
    cv2_img = bridge.imgmsg_to_cv2(data)
    cv2.imshow('ROS Image',cv2_img)
    cv2.waitKey(1)

def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('listener', anonymous=True)

    rospy.Subscriber('Avoid_Object_Position', PolygonStamped, Axiscallback)
    rospy.Subscriber('Postproc_Images', Image, Imgcallback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()
