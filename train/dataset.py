import numpy as np
import os
import cv2
from PIL import Image

from torch.utils.data import Dataset

EXTENSIONS = ['.jpg', '.png']

def load_image(file):
    return Image.open(file)

def is_image(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS)

def is_label(filename): #TODO
    #return filename.endswith("_labelTrainIds.png")
    return filename.endswith("_forFreiburgForest.png")

def image_path(root, basename, extension):
    return os.path.join(root, f'{basename}{extension}')

def image_path_city(root, name):
    # print("333333333333333333333333333",os.path.join(root, f'{name}'))
    return os.path.join(root, f'{name}')

def image_basename(filename):
    return os.path.basename(os.path.splitext(filename)[0])

class VOC12(Dataset):

    def __init__(self, root, input_transform=None, target_transform=None):
        self.images_root = os.path.join(root, 'images')
        self.labels_root = os.path.join(root, 'labels')

        self.filenames = [image_basename(f)
            for f in os.listdir(self.labels_root) if is_image(f)]
        self.filenames.sort()

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        filename = self.filenames[index]

        with open(image_path(self.images_root, filename, '.jpg'), 'rb') as f:
            image = load_image(f).convert('RGB')
        with open(image_path(self.labels_root, filename, '.png'), 'rb') as f:
            label = load_image(f).convert('P')

        if self.input_transform is not None:
            image = self.input_transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label

    def __len__(self):
        return len(self.filenames)



class cityscapes(Dataset):

    def __init__(self, root, co_transform=None, subset='train'):
        self.images_root = os.path.join(root, 'leftImg8bit/')
        self.labels_root = os.path.join(root, 'gtFine/')
        
        self.images_root += subset
        self.labels_root += subset

        #print (self.images_root)
        #self.filenames = [image_basename(f) for f in os.listdir(self.images_root) if is_image(f)]
        self.filenames = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.images_root)) for f in fn if is_image(f)]
        self.filenames.sort()

        #[os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(".")) for f in fn]
        #self.filenamesGt = [image_basename(f) for f in os.listdir(self.labels_root) if is_image(f)]
        self.filenamesGt = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.labels_root)) for f in fn if is_label(f)]
        self.filenamesGt.sort()

        self.co_transform = co_transform # ADDED THIS


    def __getitem__(self, index):
        filename = self.filenames[index]
        filenameGt = self.filenamesGt[index]
        # print("111111111111111111111111111 ", filename)
        # print("f2222222222222222222222222222222222  ",filenameGt)
        with open(image_path_city(self.images_root, filename), 'rb') as f:
            image = load_image(f).convert('RGB')
        with open(image_path_city(self.labels_root, filenameGt), 'rb') as f:
            label = load_image(f).convert('P')

        if self.co_transform is not None:
            image, label = self.co_transform(image, label)

        return image, label

    def __len__(self):
        return len(self.filenames)


class freiburgForest(Dataset):

    def __init__(self, root, co_transform=None, subset='train'):
        self.images_root = os.path.join(root, subset+'/rgb')
        self.labels_root = os.path.join(root, subset+'/GT_color')
        # print (self.images_root)
        # self.filenames = [image_basename(f) for f in os.listdir(self.images_root) if is_image(f)]
        self.filenames = [os.path.join(self.images_root, f) for f in os.listdir(self.images_root)]
        self.filenames.sort()
        self.filenamesGt = [os.path.join(self.labels_root, f) for f in os.listdir(self.labels_root)]
        self.filenamesGt.sort()

        self.co_transform = co_transform  # ADDED THIS

    def __getitem__(self, index):
        filename = self.filenames[index]
        filenameGt = self.filenamesGt[index]
        # print("111111111111111111111111111 ", filename)
        # print("f2222222222222222222222222222222222  ",filenameGt)
        with open(image_path_city(self.images_root, filename), 'rb') as f:
            image = load_image(f).convert('RGB')
        with open(image_path_city(self.labels_root, filenameGt), 'rb') as f:
            label = load_image(f).convert('L')
        # print("image:   ",image.getpixel((0,0)))
        # print("label:   ",label.getpixel((0,0)))
        if self.co_transform is not None:
            image, label = self.co_transform(image, label)
        # print("image:   ",image.size())
        # print("label:   ",label.size())
        # temp = set()
        # for i in range(512):
        #     for j in range(1024):
        #         temp.add(int(label[0][i][j].numpy()))
        # print("111111111111111111111",temp)
        return image, label

    def __len__(self):
        return len(self.filenames)

class geoMat(Dataset):

    def __init__(self, root, co_transform=None, subset='train'):
        self.images_root = os.path.join(root, subset+'/rgb')
        self.labels_root = os.path.join(root, subset+'/GT_color')
        self.filenames = []
        self.filenamesGt = []
        for dir_1 in os.listdir(self.images_root):
            temp_1= [os.path.join(self.images_root+"/"+dir_1, f) for f in os.listdir(self.images_root+"/"+dir_1) if '800x800' in f] #只选择800*800的进行训练
            self.filenames.extend(temp_1)
        self.filenames.sort()
        for dir_1 in os.listdir(self.labels_root):
            temp_2 = [os.path.join(self.labels_root+"/"+dir_1, f) for f in os.listdir(self.labels_root+"/"+dir_1) if '800x800' in f] #只选择800*800的进行训练
            self.filenamesGt.extend(temp_2)
        self.filenamesGt.sort()
        assert len(self.filenames) == len(self.filenamesGt)
        self.co_transform = co_transform  # ADDED THIS


    def __getitem__(self, index):
        filename = self.filenames[index]
        filenameGt = self.filenamesGt[index]

        image = np.array(cv2.imread(filename)).astype(np.float32)

        label = np.array(cv2.imread(filenameGt, cv2.IMREAD_GRAYSCALE)).astype(np.float32)
        if self.co_transform is not None:
            image, label = self.co_transform(image, label)

        return image, label, filename

    def __len__(self):
        return len(self.filenames)



if __name__ == "__main__":
    from train.train_for_traversability import MyCoTransform
    co_transform = MyCoTransform(augment=True)
    dataset_train = geoMat("/mrtstorage/users/pan/material_dataset_v2/", co_transform, 'train')