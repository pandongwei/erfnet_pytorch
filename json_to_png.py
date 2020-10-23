'''
该脚本文件是将做好的label文件转换为可以直接进行训练的png文件
'''
import cv2
import os

# 转换json文件到png图片
def json_to_png(images_root):
    json_paths = [os.path.join(images_root, f) for f in os.listdir(images_root) if ".json" in f]
    json_paths.sort()
    for json_path in json_paths:
        a = 'labelme_json_to_dataset '+ json_path + ' -o ' + json_path[:-5] + '.png'
        a = a.replace('e ','e\\ ')
        os.system(a)

def main():
    images_root = '/media/pandongwei/Extreme SSD/work_relative/extract_img/leftImg8bit'
    save_root = '/media/pandongwei/Extreme SSD/work_relative/extract_img/gtFine/'

    json_to_png(images_root)

    os.makedirs(os.path.dirname(save_root), exist_ok=True)
    paths_label = []
    paths_img = []
    for dir_1 in os.listdir(images_root):
        if '.png' in dir_1:
            temp_img = [os.path.join(images_root + "/" + dir_1, f) for f in os.listdir(images_root + "/" + dir_1) if "label.png" in f]
            paths_label.extend(temp_img)
        if '.jpg' in dir_1:
            temp_img = images_root + "/" + dir_1
            paths_img.append(temp_img)
    paths_img.sort()
    paths_label.sort()

    for path in paths_label:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        # label颜色更改
        ret1, img = cv2.threshold(img, 37, 255, cv2.THRESH_BINARY_INV) # 38
        save_path = save_root + path.split("extract_img/leftImg8bit/")[1].split("/label")[0][:-4]+"_labelTrainIds.png"
        cv2.imwrite(save_path, img)
    # 将图片转为png格式
    for path in paths_img:
        img = cv2.imread(path)
        #save_path = path[:-4] + '.png'
        save_path = path.split('leftImg8bit')[0] + 'RGB' + path.split('leftImg8bit')[1][:-4] +'.png'
        cv2.imwrite(save_path, img)


if __name__ == '__main__':
    main()
