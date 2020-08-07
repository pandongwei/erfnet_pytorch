import scipy.io as scio
import os
import cv2
import numpy as np

road_traversability = {
    "Asphalt": 1.0,
    "Cement - Granular": 1.0,
    "Cement - Smooth": 1.0,
    "Concrete - Precast": 1.0,
    "Foliage": 0.4,
    "Grass": 0.4,
    "Gravel": 0.2,
    "Paving": 1.0,
    "Soil - Compact": 1.0,
    "Soil - Dirt and Vegetation": 0.8,
    "Soil - Loose": 0.8,
    "Soil - Mulch": 0.6,
    "Stone - Granular": 1.0,
    "Wood": 1.0
}
hard_road = ("Asphalt", "Cement - Granular", "Cement - Smooth", "Concrete - Precast",
             "Paving", "Soil - Compact", "Stone - Granular", "Wood")

def traversability_processing(img, class_name):
    img = img.astype(np.float32)
    try:
        coef = road_traversability[class_name]
    except:
        print("can not find the corresponding road class and coef is set to 1.0")
        print(class_name)
        coef = 1.0
    img *= coef
    img = img.astype(np.uint8)
    if class_name in hard_road:
        mask = (img < 128)
        img[mask] = 128
    return img

def main():
    folder_path = "/mrtstorage/users/pan/material_dataset_v2/label/"
    path_save = "/mrtstorage/users/pan/material_dataset_v2/label_processing/"
    for dir_1 in os.listdir(folder_path):
        for dir_2 in os.listdir(folder_path+dir_1):
            file_folder_path = folder_path + dir_1+"/"+dir_2
            os.makedirs(os.path.dirname(path_save+dir_1+"/"+dir_2+"/"), exist_ok=True)
            for root, dirs, files in os.walk(file_folder_path,topdown=True):
                for file in files:
                    # 读取图片的路径
                    file_path = os.path.join(file_folder_path, file)
                    # 保存图片的路径
                    save_path = path_save+dir_1+"/"+dir_2+"/" + file

                    img = cv2.imread(file_path)

                    img = traversability_processing(img,dir_1)

                    # 对深度图做处理
                    # min_pixel,max_pixel,_,_ = cv2.minMaxLoc(img)
                    # print(min_pixel, max_pixel)
                    #depth_img = cv2.normalize(depth_img,None,alpha=0,beta=255,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8UC3)
                    cv2.imwrite(save_path, img)

if __name__ == '__main__':
    main()
