# print(2.8149201869965+6.9850029945374)
# print(3.7890393733978+9.9428062438965+9.7702074050903+9.5168733596802+10.373730659485+6.6616044044495+10.260489463806+10.405355453491+10.138095855713)
# print(4.6323022842407)
# color = [0,255,0]
# print(color[0]*0.299 + color[1]*0.587 + color[2]*0.114)
import open3d as o3d
import scipy.io as scio
import numpy as np
import cv2


dataFile1 = "/home/pan/Downloads/Grass/Train/003_001_MLB_0638_100x100_1.mat"
data1 = scio.loadmat(dataFile1)
depth_img = (data1['Depth'])
depth_img = cv2.normalize(depth_img,None,alpha=0,beta=255,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8UC3)
img = data1['Image']
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.imwrite('img.png',img)
cv2.imwrite('depth.png',depth_img)
# print(depth_img.dtype)
# depth_image_o3d = o3d.geometry.Image(img)
# intrinsic = data1['Intrinsics']
# extrinsic = data1['Extrinsics']
# temp = np.array([0,0,0,1])
# extrinsic = np.vstack((extrinsic,temp))
# dataFile = "/mrtstorage/users/pan/CVPR16_JMD_GeoInfMatRec_CodeAndData/World/data/GeoMatD/Grass/Train/003_001_MLB_0638_100x100_1_00001.mat"
# data = scio.loadmat(dataFile)
# #Vector3dVector
# pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(100,100,intrinsic[0,0],intrinsic[1,1],intrinsic[0,2],intrinsic[1,2])
#
# point = o3d.geometry.PointCloud().create_from_depth_image(depth_image_o3d,pinhole_camera_intrinsic,extrinsic)
# print(point)
# rgbd = create_rgbd_image_from_color_and_depth(color, depth, convert_rgb_to_intensity = False)
# pcd = create_point_cloud_from_rgbd_image(rgbd, pinhole_camera_intrinsic)

# flip the orientation, so it looks upright, not upside-down
# pcd.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
#
# draw_geometries([pcd])