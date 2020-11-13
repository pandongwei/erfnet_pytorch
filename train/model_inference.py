#!/usr/bin/env python3

import os
import math
import numpy as np
import torch
import cv2
import socket
import time
import struct
from argparse import ArgumentParser
from torch.autograd import Variable
from torchvision.transforms import ToTensor
from transform import Colorize
from erfnet import ERFNet
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class publisher:

    def __init__(self):
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.start1 = bytes([0xAA])
        self.start2 = bytes([0xAF])
        self.end_bs = bytes([0x66])

    def publish(self, angle):
        # m/s
        vel0 = floatToBytes(0.1)
        vel1 = floatToBytes(0.0)
        vel2 = floatToBytes(0.0)

        # rad/s
        ang_vel0 = floatToBytes(0.0)
        ang_vel1 = floatToBytes(0.0)
        ang_vel2 = floatToBytes(angle)

        msg = self.start1 + self.start2 + vel0 + vel1 + vel2 + ang_vel0 + ang_vel1 + ang_vel2 + self.end_bs  # (start1+start2+bytes_d+bytes_d+bytes_d)
        server_address = ("10.60.83.254", 1324)  # 接收方 服务器的ip地址和端口号
        self.client_socket.sendto(msg, server_address)  # 将msg内容发送给指定接收方

class image_converter:

    def __init__(self, model, out):

        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/color/image_raw",Image,self.callback)
        self.angle_pre = 0
        self.model = model
        self.out = out
        self.pub = publisher()

    def callback(self, data):
        try:
            image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            self.out.write(image)
            angle = inference(self.model, image, self.angle_pre)
            self.angle_pre = angle
            self.pub.publish(angle)
        except CvBridgeError as e:
            print(e)

def perception_to_angle(map, angle_pre, grid_size=10):
    tmp = map.copy()
    # 图片降采样
    size = map.shape
    map = cv2.resize(map, (size[1] // grid_size, size[0] // grid_size))
    map = cv2.resize(map, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)

    seed = 240
    angle, momenton = 0, 0.5
    path_point = []
    sum_point = [0,0]
    for i in range(470, -10, -10):
        if any(map[i][seed] != (255, 0, 0)): break
        sum_left, sum_right = 0, 0
        left, right = seed, seed
        while left >= 0 and all(map[i][left]==(255,0,0)):
            left -= 10
            sum_left += 10
        while right < 640 and all(map[i][right]==(255,0,0)):
            right += 10
            sum_right += 10
        seed = seed - sum_left//2 + sum_right//2
        path_point.append((seed,i))
        sum_point[0] += seed
        sum_point[1] += i

    map = tmp.copy()
    if len(path_point) > 1:
        for i in range(len(path_point)-1):
            cv2.line(map, path_point[i], path_point[i+1], (80,80,255), 3)
        cv2.line(map, path_point[0], (path_point[0][0], path_point[0][1]-240), (0,0,0), 2)
        # 角度更新的方式
        start, end = path_point[0], path_point[-1]
        #angle = math.atan2(start[0]-end[0],start[1]-end[1])

        point_average = tuple([sum_point[0] // len(path_point), sum_point[1] // len(path_point)])
        angle = math.atan2(start[0]-point_average[0], start[1]-point_average[1])

        # 加上 momenton
        angle = momenton * angle_pre + (1-momenton) * angle
        cv2.line(map, path_point[0],(path_point[0][0]-int(math.tan(angle)*100),path_point[0][1]-100),(0,0,0),3)
    return map, angle

def floatToBytes(f):
    bs = struct.pack("f",f)
    #print("pkg:",bs[0])
    #print("pkg:",bs[1])
    #print("pkg:",bs[2])
    #print("pkg:",bs[3])
    #print("b="+str(bs))
    return bs

def inference(model, image, angle_pre):
    # pre-process
    image = (image / 255.).astype(np.float32)
    image = ToTensor()(image).unsqueeze(0)
    image = image.cuda()
    input = Variable(image)
    # inference
    with torch.no_grad():
        output = model(input)
    # post-process
    label = output[0].max(0)[1].byte().cpu().data
    label_color = Colorize()(label.unsqueeze(0))
    label_save = label_color.numpy()
    label_save = label_save.transpose(1, 2, 0)
    # 加上路径规划
    label_save, angle = perception_to_angle(label_save, angle_pre)

    return angle

def talker(args):
    NUM_CLASSES = 4
    color_transform = Colorize(NUM_CLASSES)
    # Load Model
    savedir = f'../save/{args.savedir}'
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    model = ERFNet(NUM_CLASSES)
    if args.cuda:
        model = torch.nn.DataParallel(model).cuda()
    model_dir = args.model_dir
    def load_my_state_dict(model, state_dict):
        # state_dict = state_dict["state_dict"]
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            own_state[name].copy_(param)
        return model
    model = load_my_state_dict(model, torch.load(model_dir))
    model.eval()
    # parameters about saving video
    video_save_path = args.video_save_path
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_save_path + 'output_new.avi', fourcc, 10.0, (640, 480))

    ic = image_converter(model, out)
    rospy.init_node('talker', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    out.release()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cuda', action='store_true', default=True)  #NOTE: cpu-only has not been tested so you might have to change code if you deactivate this flag
    parser.add_argument('--model', default="erfnet")
    parser.add_argument('--state')
    parser.add_argument('--port', type=int, default=8097)
    parser.add_argument('--savedir', default="cityscape_4class_1")
    parser.add_argument('--datadir', default="/media/pandongwei/Extreme SSD/work_relative/extract_img/")
    parser.add_argument('--video_save_path', default="/home/pandongwei/work_repository/erfnet_pytorch/eval/")
    parser.add_argument('--model_dir', default="/home/pandongwei/work_repository/erfnet_pytorch/save/" + "cityscape_4class_1" + '/model_best.pth')
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--num-epochs', type=int, default=150) #150
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--steps-loss', type=int, default=50)
    parser.add_argument('--steps-plot', type=int, default=50)
    parser.add_argument('--epochs-save', type=int, default=0)    #You can use this value to save model every X epochs
    parser.add_argument('--decoder', action='store_true',default=True)
    parser.add_argument('--pretrainedEncoder', default="../trained_models/erfnet_encoder_pretrained.pth.tar")
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--iouTrain', action='store_true', default=False) #recommended: False (takes more time to train otherwise)
    parser.add_argument('--iouVal', action='store_true', default=True)  
    parser.add_argument('--resume', action='store_true')    #Use this flag to load last checkpoint for training
    talker(parser.parse_args())
