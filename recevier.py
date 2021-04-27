#!/usr/bin/env python
# Software License Agreement (BSD License)
#
# Copyright (c) 2008, Willow Garage, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of Willow Garage, Inc. nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Revision $Id$

## Simple talker demo that listens to std_msgs/Strings published 
## to the 'chatter' topic

import rospy,re,argparse
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import PolygonStamped,Point32

import cv2,socket,numpy,time,select
from multiprocessing import Process

class rospublisher:
    """
    @description  : handle relevant events of ROS topic publishing 
    ---------
    @function  :
    -------
    """
    count = 0
    def __init__(self):
        rospublisher.count += 1
        rospy.init_node('AvoidModel', anonymous=True)
        self._coor_pub=rospy.Publisher('Avoid_Object_Position',PolygonStamped,queue_size=10)    
        self._img_pub=rospy.Publisher('Postproc_Images',Image,queue_size=10)
        self.r=rospy.Rate(1)

    def __del__(self):
        rospublisher.count -= 1
        print ("RosPublisher release")

    def pub_axis(self,coords,frame,num):
        """
        @description  : publisher the coordinates of the avoid object to ROS 
        ---------
        @param  :
            coords: type list, list of coordinates 
            frame: type int, image frame no.
            num: type int, the number of object in coordinates
        -------
        @Returns  : None
        -------
        """
        
        
        if coords != []:
            coord_msg=PolygonStamped()
            coord_msg.header.stamp=rospy.Time.now()
            coord_msg.header.frame_id=str(frame)
            for i in range(int(num)):
                coord_msg.polygon.points.append(Point32())
                coord_msg.polygon.points[4*i].x=float(coords[i*6+2])/1000
                coord_msg.polygon.points[4*i].y=float(-float(coords[i*6+0])/1000)
                coord_msg.polygon.points[4*i].z=float(coords[i*6+1])/1000
                coord_msg.polygon.points.append(Point32())
                coord_msg.polygon.points[4*i+1].x=float(coords[i*6+2])/1000+0.2
                coord_msg.polygon.points[4*i+1].y=float(-float(coords[i*6+0])/1000)
                coord_msg.polygon.points[4*i+1].z=float(coords[i*6+1])/1000
                coord_msg.polygon.points.append(Point32())
                coord_msg.polygon.points[4*i+2].x=float(coords[i*6+5])/1000
                coord_msg.polygon.points[4*i+2].y=float(-float(coords[i*6+3])/1000)
                coord_msg.polygon.points[4*i+2].z=float(coords[i*6+4])/1000
                coord_msg.polygon.points.append(Point32())
                coord_msg.polygon.points[4*i+3].x=float(coords[i*6+5])/1000+0.2
                coord_msg.polygon.points[4*i+3].y=float(-float(coords[i*6+3])/1000)
                coord_msg.polygon.points[4*i+3].z=float(coords[i*6+4])/1000
            
            self._coor_pub.publish(coord_msg)
            
    def pub_img(self,img):
        bridge=CvBridge()
        self._img_pub.publish(bridge.cv2_to_imgmsg(img, encoding="passthrough"))

# 接受图片及大小的信息
def recvall(sock, count):#读取count长度的数据
    """
    @description  : socket server receive an fixed length packet
    ---------
    @param  :
        sock: socket server connector handler
        count: type int, length of the packet to receive
    -------
    @Returns  : buf, type bytes*count, the received data 
    -------
    """
    
    buf = b''
    while count:
        newbuf = sock.recv(count)  # s.sendall()发送, s.recv()接收. 因为client是通过 sk.recv()来进行接受数据，而count表示，最多每次接受count字节，
        # print(newbuf)
        if not newbuf: return buf
        buf += newbuf
        count -= len(newbuf)
    return buf

def netdata_pipe(server_soc, videoWriter, pub):
    """
    @description  : recevie image from socket server
    ---------
    @param  :
        conn: the handler of the socket server
        addr: address of socket server
        pub: the handler of the ROS node
    -------
    @Returns  : None
    -------
    """  

    # connected_clients_sockets = []
    # connected_clients_sockets.append(server_soc) 
    block=1024 #1280*720
    conn, client_address = server_soc.accept()
    print('connect from:' + str(client_address))
    while True:
        start = time.time()
        try:
            # read_sockets, _, _ = select.select(connected_clients_sockets, [], [])
            # for sock in read_sockets:
            #     if sock == server_soc:
                # connected_clients_sockets.append(conn)
            # conn.settimeout(0.001)
            image_size=0
            coords=[]
            img_flag = False
            
            stringData = conn.recv(block)#nt()只能转化由纯数字组成的字符串
            if stringData != b'':
                # print('check point 1',stringData.startswith(b'$Image'))
                # try:
                if stringData.startswith(b'$Image'):
                    stringData = stringData.decode()
                    image_size=int(stringData.split(',')[1])
                    # print(stringData)
                    coords = []
                    if opt.image:
                        conn.sendall('Ready for Image'.encode('utf-8'))
                        stringData = recvall(conn, (image_size))#nt()只能转化由纯数字组成的字符串
                        # print('check point 2',stringData == b'')
                        img = numpy.frombuffer(stringData,numpy.uint8)  # 将获取到的字符流数据转换成1维数组 data = numpy.fromstring()           numpy.frombuffer
                        decimg = cv2.imdecode(img, cv2.IMREAD_COLOR)  # 将数组解码成图像
                        pub.pub_img(decimg)
                        
                    # if opt.image:
                    #     cv2.imshow('SERVER', decimg)  # 显示图像
                    #     if cv2.waitKey(1) == ord('q'):
                    #         break 
                    # videoWriter.write(decimg)
                    conn.sendall('Ready for Coordinates'.encode('utf-8'))
                    stringData = conn.recv(block)#nt()只能转化由纯数字组成的字符串
                    # print('check point 3',stringData == b'')
                    stringData = stringData.decode()
                    # print(stringData)
                    coords=stringData.split(',')[1:-1]
                    # for i in range(len(coords)):
                    #     print(coords[i])
                    assert len(coords) % 6 == 0,'coords length error'
                    conn.sendall('Ready for next Frame'.encode('utf-8'))
                    pub.pub_axis(coords,1,(len(coords)/6))
                    # =================================================================================================================================  
                    time.sleep(0.05)
                    print('process time = ', (time.time()-start))
                    # =================================================================================================================================
                else:
                    continue
        except KeyboardInterrupt:
            if conn:
                conn.close()  
            break    
    print('network done')




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=9191, help='tcp port')
    parser.add_argument('--ip', type=str, default='192.168.3.181', help='tcp IP')
    parser.add_argument('--image', action='store_true', help='show image mode')
    opt = parser.parse_args()
    pub=rospublisher()
    address = (opt.ip, opt.port)#'192.168.1.104', 8004
    server_soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_soc.bind(address)
    server_soc.listen(10)
    # fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I','D')#avi格式
    # videoWriter = cv2.VideoWriter('./runs/receive_video/%s.avi' % time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time())) , fourcc, 1, (1280, 720))##%(str(time.strftime("%Y-%m-%d-%H:%M:%S",time.localtime())))
    videoWriter = []
    
    netdata_pipe(server_soc,videoWriter,pub)

    # p=Process(target=ReceiveVideo,args=(conn, addr, pub))  #daemon默认值为False，如果设置为True，代表该进程为后台守护进程；当该进程的父进程终止时，该进程也随之终止；并且设置为True后，该进程不能创建子进程，设置该属性必须在start()之前
    # p.start()