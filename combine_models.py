'''
在完成模型训练之后，进行结果合并
author:dongwei pan
'''

import os
import random
import time
import json
import numpy as np
import cv2
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from train.dataset import freiburgForest1
from train.transform import Colorize
from train.erfnet import ERFNet
from train.multi_models import Multi_models
from path_planning import path_planning


def colormap_cityscapes(n):
    # 默认为RGB形式
    cmap = np.zeros([n+1, 3]).astype(np.uint8)
    for i in range(n):
        cmap[i, :] = np.array([255-i, i, 0])
    cmap[n,:] = np.array([0, 0, 255])
    return cmap


class Colorize:
    # BGR的形式
    def __init__(self):
        self.cmap = colormap_cityscapes(256)
        # self.cmap[n] = self.cmap[-1]

    def __call__(self, gray_image):
        shape = gray_image.shape
        #print(size)
        color_image = np.zeros([3, shape[0],shape[1]])
        # print(gray_image.shape, color_image.shape)
        for label in range(0, len(self.cmap)):
            mask = gray_image[:,:,0] == label
            is_sky = gray_image[:,:,0] < 0
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]
            color_image[0][is_sky] = self.cmap[label][0]
            color_image[1][is_sky] = self.cmap[label][1]
            color_image[2][is_sky] = self.cmap[label][2]

        return color_image


class MyCoTransform(object):
    def __init__(self, augment=True, rescale=True, size=(104,104)):

        self.augment = augment
        self.size = size
        self.rescale = rescale

    def __call__(self, input, target):

        input = cv2.resize(input,self.size,interpolation=cv2.INTER_LINEAR)
        target = cv2.resize(target,self.size,interpolation=cv2.INTER_NEAREST)

        if self.rescale:
            input = input/255.
            target = target/255.

        if(self.augment):
            # Random hflip
            hflip = random.random()
            if (hflip < 0.5):
                # input = input.transpose(Image.FLIP_LEFT_RIGHT)
                # target = target.transpose(Image.FLIP_LEFT_RIGHT)
                input = cv2.flip(input,1)
                target = cv2.flip(target,1)

        input = ToTensor()(input)
        target = torch.from_numpy(np.array(target)).long().unsqueeze(0)

        return input, target


# build the loss function nagetive Gaussian log-likelihood loss
class GaussianLogLikelihoodLoss(torch.nn.Module):
    def __init__(self):
        super(GaussianLogLikelihoodLoss,self).__init__()

    def forward(self, outputs, targets):
        mean, var = outputs[:,0,:,:],outputs[:,1,:,:]
        loss_1 = torch.mean(torch.pow(mean - targets,2)/(2*torch.pow(var,2)))
        loss_2 = torch.mean(torch.log(var))
        # print("loss1: ",loss_1)
        # print('loss2: ',loss_2)
        return 0.5*(loss_1 + 0.01*loss_2)

# 模型结果测试
def test(model_geoMat, model_freiburgForest, dataloader_test, cfg):
    cuda = cfg['cuda']
    savedir = cfg['savedir']
    data_savedir = f'eval/{savedir}'
    model_geoMat.eval()
    model_freiburgForest.eval()

    coef = np.ones([512, 1024, 5]).astype(np.float32)
    traversability = (0,1.0,0.6,0.8,-1)
    for i in range(5):
        coef[:,:,i] = coef[:,:,i]*traversability[i]

    # traversability = {"0": 0, "1": 1, "2": 0.6, "3": 0.8, "4":-1}
    for step, (images, labels, filename) in enumerate(dataloader_test):
        if (cuda):
            images = images.cuda()

        inputs = Variable(images)

        start_time = time.time()
        with torch.no_grad():
            outputs_1 = model_freiburgForest(inputs)
            outputs_2 = model_geoMat(inputs)
            outputs_2 = outputs_2[0]   # TODO
        # batch size = 1,结果处理成可以显示的格式
        # print(inputs.shape)
        print("model inference time: %.2f s" % (time.time() - start_time))
        start_time = time.time()
        outputs_1 = outputs_1[0].max(0)[1].byte().cpu().data.unsqueeze(0)
        # print(outputs_1.shape)
        outputs_1 = outputs_1.numpy().transpose(1, 2, 0).astype(np.int64)
        # print(outputs_1.shape)
        outputs_1 = np.take(coef, outputs_1)

        outputs_2 = outputs_2[0, 0, :, :].cpu().data.unsqueeze(0)
        outputs_2 = outputs_2.numpy().transpose(1, 2, 0)

        outputs_combine = (outputs_1*outputs_2*255).astype(np.int16)
        # min_pixel, max_pixel, _, _ = cv2.minMaxLoc(outputs_combine)
        # print(min_pixel,'   ', max_pixel)
        # outputs_2 = cv2.normalize(outputs_2, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        # outputs_2 = outputs_2[:,:,np.newaxis]

        print("output processing time 1: %.2f s" % (time.time() - start_time))

        start_time = time.time()
        outputs_combine = Colorize()(outputs_combine).transpose(1, 2, 0).astype(np.float32)
        print("colorize time: %.2f s" % (time.time() - start_time))
        images = images.cpu().numpy()
        images = images[0].transpose(1, 2, 0)*255

        # print(output_combine.shape, images.shape)
        fileSave = data_savedir + filename[0].split("freiburg_forest_annotated")[1]
        os.makedirs(os.path.dirname(fileSave), exist_ok=True)
        # output = cv2.addWeighted(images, 0.4, output_combine, 0.6, 0)
        # min_pixel, max_pixel, _, _ = cv2.minMaxLoc(output)
        # print(min_pixel,'   ', max_pixel)
        cv2.imwrite(fileSave,outputs_combine)


# 测试单张图片
def test_img(model_geoMat, model_freiburgForest, cfg):
    img_path = "/home/pandongwei/work_repository/erfnet_pytorch/test_img/row.png"

    cuda = cfg['cuda']
    model_geoMat.eval()
    model_freiburgForest.eval()

    coef = np.ones([512, 1024, 5]).astype(np.float32)
    traversability = (0, 1.0, 0.6, 0.8, -1)
    for i in range(5):
        coef[:, :, i] = coef[:, :, i] * traversability[i]

    image = cv2.imread(img_path).astype(np.float32)
    image = cv2.resize(image,(1024,512),interpolation=cv2.INTER_LINEAR)
    image = image/255.
    image = ToTensor()(image).unsqueeze(0)
    if (cuda):
        image = image.cuda()

    input = Variable(image)

    # start_time = time.time()
    with torch.no_grad():
        outputs_1 = model_freiburgForest(input)
        outputs_2 = model_geoMat(input)
        outputs_2 = outputs_2[0]  # TODO

    #print("model inference time: %.2f s" % (time.time() - start_time))
    # start_time = time.time()
    outputs_1 = outputs_1[0].max(0)[1].byte().cpu().data.unsqueeze(0)
    outputs_1 = outputs_1.numpy().transpose(1, 2, 0).astype(np.int64)
    outputs_1 = np.take(coef, outputs_1)

    outputs_2 = outputs_2[0, 0, :, :].cpu().data.unsqueeze(0)
    # print(outputs_2.shape)
    outputs_2 = outputs_2.numpy().transpose(1, 2, 0)

    outputs_combine = (outputs_1 * outputs_2 * 255).astype(np.int16)
    cv2.imwrite('test.png', outputs_combine)


# 模型推导，结果合并，并进行路径规划和落脚点规划
def save_video_and_path_planning(model_geoMat, model_freiburgForest, cfg):
    image_folder = cfg["image_folder_raw"]
    video_save_path = "/home/pandongwei/work_repository/erfnet_pytorch/eval/"

    # parameters about saving video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_save_path+'output.avi', fourcc, 10.0, (640, 480))

    cuda = cfg['cuda']
    model_geoMat.eval()
    model_freiburgForest.eval()

    coef = np.ones([512, 1024, 5]).astype(np.float32)
    traversability = (0, 1.0, 0.6, 0.8, -1)
    for i in range(5):
        coef[:, :, i] = coef[:, :, i] * traversability[i]

    paths = []
    for root, dirs, files in os.walk(image_folder, topdown=True):
        for file in files:
            image_path = os.path.join(image_folder, file)
            paths.append(image_path)
    paths.sort()
    font = cv2.FONT_HERSHEY_SIMPLEX

    for i, path in enumerate(paths):
        start_time = time.time()
        image = cv2.imread(path)
        image = cv2.resize(image,(512,256),interpolation=cv2.INTER_LINEAR)
        image = image/255.
        image = ToTensor()(image).unsqueeze(0)
        if (cuda):
            image = image.cuda()

        input = Variable(image)

        with torch.no_grad():
            outputs_1 = model_freiburgForest(input)
            outputs_2 = model_geoMat(input)
            outputs_2 = outputs_2[0]  # TODO

        outputs_1 = outputs_1[0].max(0)[1].byte().cpu().data.unsqueeze(0)
        # print(outputs_1.shape)
        outputs_1 = outputs_1.numpy().transpose(1, 2, 0).astype(np.int64)
        # print(outputs_1.shape)
        outputs_1 = np.take(coef, outputs_1)

        outputs_2 = outputs_2[0, 0, :, :].cpu().data.unsqueeze(0)
        # print(outputs_2.shape)
        outputs_2 = outputs_2.numpy().transpose(1, 2, 0)

        # 加上momenton来稳定算法 TODO
        if i==0:
            momenton = 0.9
            output_pre = outputs_2.copy()
        else:
            outputs_2 = (momenton*output_pre + (1-momenton)*outputs_2)
            output_pre = outputs_2.copy()

        outputs_combine = (outputs_1 * outputs_2 * 255).astype(np.uint8)


        outputs_combine_color = Colorize()(outputs_combine).transpose(1, 2, 0).astype(np.uint8)
        outputs_combine_color = cv2.cvtColor(outputs_combine_color, cv2.COLOR_RGB2BGR)


        image = image.cpu().numpy()
        image = (image[0].transpose(1, 2, 0) * 255).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # print(outputs_combine.shape)
        # output = np.hstack([outputs_combine_color, image])
        # print(" post process time: %.2f s" % (time.time() - start_time))
        # start_time = time.time()

        outputs_combine = outputs_combine[:,:,0]
        output = path_planning(outputs_combine, outputs_combine_color)

        # print(" path_planning time: %.2f s" % (time.time() - start_time))
        # start_time = time.time()

        output = np.hstack([output, image])
        cv2.putText(output, str(round(1/(time.time() - start_time), 1))+" Hz",(50,50), font, 1, (0, 0, 255), 2)
        out.write(image)

        print(i, "  time: %.2f s" % (time.time() - start_time))
    out.release()

def main():
    # load parameters from the config file
    with open('config/config.json', 'r') as f:
        cfg = json.load(f)
    batch_size = int(cfg['batch_size'])
    cuda = cfg['cuda']
    model = cfg['model']
    size_geoMat = int(cfg['size_geoMat'])
    width_freiburgForest = int(cfg['width_freiburgForest'])
    height_freiburgForest = int(cfg['height_freiburgforest'])
    epochs = int(cfg['epochs'])
    learning_rate = float(cfg['learning_rate'])
    num_workers = int(cfg['num_workers'])
    steps_loss = int(cfg['steps_loss'])
    epochs_save = int(cfg['epochs_save'])
    datadir_geoMat = cfg['datadir_geoMat']
    datadir_freiburgForest = cfg['datadir_freiburgForest']
    savedir = cfg['savedir']
    model_geoMat = cfg['model_geoMat']
    weight_geoMat = cfg['weight_geoMat']
    model_freiburgForest = cfg['model_freiburgForest']
    weight_freigburgForest = cfg['weight_freigburgForest']
    size_freiburgForest = (width_freiburgForest, height_freiburgForest)
    size_geoMat = (size_geoMat, size_geoMat)
    data_savedir = f'eval/{savedir}'
    model_savedir = f'save/{savedir}'
    if not os.path.exists(data_savedir):
        os.makedirs(data_savedir)
    if not os.path.exists(model_savedir):
        os.makedirs(model_savedir)

    #Load both Models
    model_geoMat = Multi_models()
    model_freiburgForest = ERFNet(5)
    
    if cuda:
        model_geoMat = torch.nn.DataParallel(model_geoMat).cuda()
        model_freiburgForest = torch.nn.DataParallel(model_freiburgForest).cuda()
    assert os.path.exists(datadir_geoMat) or os.path.exists(datadir_freiburgForest), "Error: datadir (dataset directory) could not be loaded"

    def load_my_state_dict(model, state_dict):
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                 continue
            own_state[name].copy_(param)
        return model
    def load_my_checkpoint(model, state_dict):
        # state_dict = state_dict["state_dict"]
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                 continue
            own_state[name].copy_(param)
        return model
    model_geoMat = load_my_checkpoint(model_geoMat, torch.load(weight_geoMat))
    model_freiburgForest = load_my_state_dict(model_freiburgForest, torch.load(weight_freigburgForest))

    print("========== START TESTING ==============")
    # co_transform_val = MyCoTransform(augment=False, rescale=True, size=size_freiburgForest)
    # dataset_val = freiburgForest1(datadir_freiburgForest, co_transform_val, 'test')
    # loader_test = DataLoader(dataset_val, num_workers=num_workers, batch_size=1, shuffle=False)
    # test(model_geoMat, model_freiburgForest, loader_test, cfg)

    # save_video(model_geoMat, model_freiburgForest, cfg)

    # test_img(model_geoMat, model_freiburgForest, cfg)

    save_video_and_path_planning(model_geoMat, model_freiburgForest, cfg)



if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  #指定使用哪张显卡  TODO
    main()
