import scipy.io as scio
import os
import cv2

def main():
    folder_path = "/mrtstorage/users/pan/materials_dataset/"
    path_save = "/mrtstorage/users/pan/material_dataset_v1/depth/"
    for dir_1 in os.listdir(folder_path):
        for dir_2 in os.listdir(folder_path+dir_1):
            file_folder_path = folder_path + dir_1+"/"+dir_2
            # os.makedirs(os.path.dirname(path_save+dir_1+"/"+dir_2+"/"), exist_ok=True)
            for root, dirs, files in os.walk(file_folder_path,topdown=True):
                for file in files:
                    # 读取图片的路径
                    file_path = os.path.join(file_folder_path, file)
                    # 保存图片的路径
                    save_path = path_save+dir_1+"/"+dir_2+"/" + file[:-4] + '.png'
                    data1 = scio.loadmat(file_path)
                    depth_img = (data1['Depth'])
                    # 对深度图做处理
                    min_pixel,_,_,_ = cv2.minMaxLoc(depth_img)
                    depth_img = (depth_img-min_pixel) # 换成正数
                    #depth_img = cv2.normalize(depth_img,None,alpha=0,beta=255,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8UC3)
                    # img = data1['Image']
                    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    # cv2.imwrite('img.png',img)
                    cv2.imwrite(save_path, depth_img)

if __name__ == '__main__':
    main()
