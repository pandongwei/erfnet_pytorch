import os
import random
import time
import torch
import json
import numpy as np
import cv2

from torch.optim import SGD, Adam, lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from tensorboardX import SummaryWriter
from train.dataset import freiburgForest, freiburgForest1
from train.transform import Colorize, Relabel
from train.erfnet import ERFNet, ERFNet_regression,ERFNet_regression_simplified
from train.iouEval import iouEval, getColorEntry
from train.multi_models import Multi_models
from shutil import copyfile

num_class = 2

def colormap_cityscapes(n):
    cmap = np.zeros([n+1, 3]).astype(np.uint8)
    for i in range(n):
        cmap[i, :] = np.array([0, i, 255-i])
    cmap[n,:] = np.array([255, 0, 0])
    return cmap

class Colorize:
    # BGR的形式
    def __init__(self, n=256):
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

#Augmentations - different function implemented to perform random augments on both image and target
class MyCoTransform(object):
    def __init__(self, augment=True, rescale=True, size=(104,104)):

        self.augment = augment
        self.size = size
        self.rescale = rescale

    def __call__(self, input, target):
        # do something to both images
        # input = Resize(self.size, Image.BILINEAR)(input)
        # target = Resize(self.size, Image.NEAREST)(target)

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
        # print(mean)
        # print(var)
        # print(targets)
        # print(torch.pow(mean - targets,2))
        # print(2*torch.pow(var,2))
        loss_1 = torch.mean(torch.pow(mean - targets,2)/(2*torch.pow(var,2)))
        loss_2 = torch.mean(torch.log(var))
        # print("loss1: ",loss_1)
        # print('loss2: ',loss_2)
        return 0.5*(loss_1 + 0.01*loss_2)


def train(model, dataloader_train, dataloader_eval, criterion, optimizer, cfg):
    cuda = cfg['cuda']
    savedir = cfg['savedir']
    resume = cfg['resume']
    train_savedir = f'save/{savedir}'
    epochs = int(cfg['epochs'])
    steps_loss = int(cfg['steps_loss'])

    min_loss = float('inf')
    # use tensorboard
    writer = SummaryWriter(log_dir=train_savedir)

    automated_log_path = train_savedir + "/automated_log.txt"
    modeltxtpath = train_savedir + "/model.txt"

    if (not os.path.exists(automated_log_path)):    #dont add first line if it exists 
        with open(automated_log_path, "a") as myfile:
            myfile.write("Epoch\t\tTrain-loss\t\tTest-loss\t\tTrain-IoU\t\tTest-IoU\t\tlearningRate")

    with open(modeltxtpath, "w") as myfile:
        myfile.write(str(model))

    start_epoch = 1
    if resume:
        #Must load weights, optimizer, epoch and best value.

        filenameCheckpoint = train_savedir + '/checkpoint.pth.tar'
        assert os.path.exists(filenameCheckpoint), "Error: resume option was used but checkpoint was not found in folder"
        checkpoint = torch.load(filenameCheckpoint)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_acc = checkpoint['best_acc']
        print("=> Loaded checkpoint at epoch {})".format(checkpoint['epoch']))

    #scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5) # set up scheduler     ## scheduler 1
    lambda1 = lambda epoch: pow((1-((epoch-1)/epochs)),0.9)  ## scheduler 2
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)                             ## scheduler 2



    for epoch in range(start_epoch, epochs+1):
        print("----- TRAINING - EPOCH", epoch, "-----")

        scheduler.step(epoch)    ## scheduler 2

        epoch_loss = []
        time_train = []

        doIouTrain = cfg['iouTrain']
        doIouVal =  cfg['iouVal']

        if (doIouTrain):
            iouEvalTrain = iouEval(num_class)

        usedLr = 0
        for param_group in optimizer.param_groups:
            print("LEARNING RATE: ", param_group['lr'])
            usedLr = float(param_group['lr'])

        model.train()
        for step, (images, labels,_) in enumerate(dataloader_train):

            start_time = time.time()
            #print (labels.size())
            #print (np.unique(labels.numpy()))
            #print("labels: ", np.unique(labels[0].numpy()))
            #labels = torch.ones(4, 1, 512, 1024).long()
            if cuda:
                images = images.cuda()
                labels = labels.cuda()
            #print("image: ", images.size())
            #print("labels: ", labels.size())
            inputs = Variable(images)
            targets = Variable(labels)
            outputs = model(inputs)

            # print("output: ", outputs.size()) #TODO
            # print("targets", np.unique(targets[:, 0].cpu().data.numpy()))

            optimizer.zero_grad()
            loss = criterion(outputs, targets[:, 0])


            loss.backward()
            optimizer.step()

            epoch_loss.append(loss)
            time_train.append(time.time() - start_time)

            if (doIouTrain):
                #start_time_iou = time.time()
                iouEvalTrain.addBatch(outputs.max(1)[1].unsqueeze(1).data, targets.data)
                #print ("Time to add confusion matrix: ", time.time() - start_time_iou)
            
        average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)
        writer.add_scalar('train_loss', average_epoch_loss_train, epoch)

        iouTrain = 0
        if (doIouTrain):
            iouTrain, iou_classes = iouEvalTrain.getIoU()
            iouStr = getColorEntry(iouTrain)+'{:0.2f}'.format(iouTrain*100) + '\033[0m'
            print ("EPOCH IoU on TRAIN set: ", iouStr, "%")  

        #Validate on 500 val images after each epoch of training
        print("----- VALIDATING - EPOCH", epoch, "-----")
        model.eval()
        epoch_loss_val = []
        time_val = []

        if (doIouVal):
            iouEvalVal = iouEval(num_class)

        for step, (images, labels,_) in enumerate(dataloader_eval):
            start_time = time.time()
            if cuda:
                images = images.cuda()
                labels = labels.cuda()
            optimizer.zero_grad()
            inputs = Variable(images)
            targets = Variable(labels)
            with torch.no_grad():
                outputs = model(inputs)

                loss = criterion(outputs, targets[:, 0])
            epoch_loss_val.append(loss.data)
            time_val.append(time.time() - start_time)

            if steps_loss > 0 and step % steps_loss == 0:
                average = sum(epoch_loss_val) / len(epoch_loss_val)
                print(f'VAL loss: {average:0.4} (epoch: {epoch}, step: {step})', 
                        "// Avg time/img: %.4f s" % (sum(time_val) / len(time_val) / cfg['batch_size']))

        average_epoch_loss_val = sum(epoch_loss_val) / len(epoch_loss_val)
        #scheduler.step(average_epoch_loss_val, epoch)  ## scheduler 1   # update lr if needed
        writer.add_scalar('train_loss', average_epoch_loss_val, epoch)

        iouVal = 0
        if (doIouVal):
            iouVal, iou_classes = iouEvalVal.getIoU()
            iouStr = getColorEntry(iouVal)+'{:0.2f}'.format(iouVal*100) + '\033[0m'
            print ("EPOCH IoU on VAL set: ", iouStr, "%") 

        is_best = average_epoch_loss_val < min_loss
        min_loss = min(min_loss, average_epoch_loss_val)

        filenameCheckpoint = train_savedir + '/checkpoint.pth.tar'
        filenameBest = train_savedir + '/model_best.pth.tar'
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': str(model),
            'state_dict': model.state_dict(),
            'best_acc': min_loss,
            'optimizer' : optimizer.state_dict(),
        }, is_best, filenameCheckpoint, filenameBest)

        #SAVE MODEL AFTER EPOCH
        filename = f'{train_savedir}/model-{epoch:03}.pth'
        filenamebest = f'{train_savedir}/model_best.pth'
        if cfg['epochs_save'] > 0 and step > 0 and step % cfg['epochs_save'] == 0:
            torch.save(model.state_dict(), filename)
            print(f'save: {filename} (epoch: {epoch})')
        if (is_best):
            torch.save(model.state_dict(), filenamebest)
            print(f'save: {filenamebest} (epoch: {epoch})')

            with open(train_savedir + "/best.txt", "w") as myfile:
                myfile.write("Best epoch is %d, with Val-IoU= %.4f" % (epoch, iouVal))

        #SAVE TO FILE A ROW WITH THE EPOCH RESULT (train loss, val loss, train IoU, val IoU)
        #Epoch		Train-loss		Test-loss	Train-IoU	Test-IoU		learningRate
        with open(automated_log_path, "a") as myfile:
            myfile.write("\n%d\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.8f" % (epoch, average_epoch_loss_train, average_epoch_loss_val, iouTrain, iouVal, usedLr ))
    writer.close()
    return(model)   #return model (convenience for encoder-decoder training)


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
        # print(outputs_2.shape)
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

def save_video(model_geoMat, model_freiburgForest, cfg):
    image_folder = "/mrtstorage/users/pan/freiburg_forest_multispectral_annotated/raw_data/freiburg_forest_raw/2016-02-26-15-05-05/"
    video_save_path = "/home/pan/repository/erfnet_pytorch/eval/"
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_save_path+'output_combine.avi', fourcc, 30.0, (1024, 512))

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
    paths.sort(key = lambda x: int(x[111:-4]))

    for i, path in enumerate(paths):
        # print(path)
        if i < 7500:
            continue
        start_time = time.time()

        image = cv2.imread(path).astype(np.float32)
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
        # print(outputs_1.shape)
        outputs_1 = outputs_1.numpy().transpose(1, 2, 0).astype(np.int64)
        # print(outputs_1.shape)

        outputs_1 = np.take(coef, outputs_1)

        outputs_2 = outputs_2[0, 0, :, :].cpu().data.unsqueeze(0)
        # print(outputs_2.shape)
        outputs_2 = outputs_2.numpy().transpose(1, 2, 0)

        outputs_combine = (outputs_1 * outputs_2 * 255).astype(np.int16)
        # min_pixel, max_pixel, _, _ = cv2.minMaxLoc(outputs_combine)
        # print(min_pixel,'   ', max_pixel)

        #print("output processing time 1: %.2f s" % (time.time() - start_time))

        # start_time = time.time()
        outputs_combine = Colorize()(outputs_combine).transpose(1, 2, 0).astype(np.uint8)

        image = image.cpu().numpy()
        image = (image[0].transpose(1, 2, 0)*255).astype(np.uint8)
        # print(image.shape)
        # print(outputs_combine.shape)
        output = cv2.addWeighted(image, 0.4, outputs_combine, 0.6, 0)
        # output = np.hstack([image, outputs_combine])
        print("time: %.2f s" % (time.time() - start_time))
        out.write(output)
    out.release()

def save_checkpoint(state, is_best, filenameCheckpoint, filenameBest):
    torch.save(state, filenameCheckpoint)
    if is_best:
        print("Saving model as best")
        torch.save(state, filenameBest)

def main():
    # load parameters
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
    # model_geoMat = ERFNet_regression_simplified()  # TODO
    model_geoMat = Multi_models()
    model_freiburgForest = ERFNet(5)
    # copyfile(args.model + ".py", savedir + '/' + args.model + ".py") # 在训练时候可以加上
    
    if cuda:
        model_geoMat = torch.nn.DataParallel(model_geoMat).cuda()
        model_freiburgForest = torch.nn.DataParallel(model_freiburgForest).cuda()

    assert os.path.exists(datadir_geoMat) or os.path.exists(datadir_freiburgForest), "Error: datadir (dataset directory) could not be loaded"

    co_transform = MyCoTransform(augment=True, rescale=False, size=size_freiburgForest)
    co_transform_val = MyCoTransform(augment=False, rescale=True, size=size_freiburgForest)
    dataset_train = freiburgForest1(datadir_freiburgForest, co_transform, 'train')
    dataset_val = freiburgForest1(datadir_freiburgForest, co_transform_val, 'test')

    loader_train = DataLoader(dataset_train, num_workers=num_workers, batch_size=batch_size, shuffle=True)
    loader_val = DataLoader(dataset_val, num_workers=num_workers, batch_size=batch_size, shuffle=False)

    criterion = GaussianLogLikelihoodLoss()
    #optimizer = Adam(model.parameters(), 5e-4, (0.9, 0.999),  eps=1e-08, weight_decay=2e-4)     ## scheduler 1
    optimizer = Adam(model_freiburgForest.parameters(), 5e-4, (0.9, 0.999),  eps=1e-08, weight_decay=1e-4)

    # 训练freiburg forest数据集的网络
    #model_freiburgForest = train(model_freiburgForest, loader_train, loader_val, criterion, optimizer, cfg)   #Train decoder
    print("========== TRAINING FINISHED ===========")

    print("========== START TESTING ==============")
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


    loader_test = DataLoader(dataset_val, num_workers=num_workers, batch_size=1, shuffle=False)
    # test(model_geoMat, model_freiburgForest, loader_test, cfg)
    save_video(model_geoMat, model_freiburgForest, cfg)



if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  ## TODO
    main()
