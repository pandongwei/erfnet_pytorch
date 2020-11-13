# Main code for training ERFNet model in Cityscapes dataset
# Sept 2017
# Eduardo Romera
#######################

import os
import random
import math
import time
import numpy as np
import torch
import cv2
from argparse import ArgumentParser

from torch.optim import SGD, Adam, lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, ToPILImage

from dataset import cityscapes_cv,gardenscapes
from transform import Relabel, ToLabel, Colorize
from visualize import Dashboard
from erfnet import ERFNet
from iouEval import iouEval, getColorEntry
from shutil import copyfile

NUM_CHANNELS = 3
NUM_CLASSES = 4 #pascal=22, cityscapes=20

color_transform = Colorize(NUM_CLASSES)
image_transform = ToPILImage()

#Augmentations - different function implemented to perform random augments on both image and target
class MyCoTransform(object):
    def __init__(self, augment=True, rescale=True, width=640, height=480):
        self.augment = augment
        self.size = (width, height)
        self.rescale = rescale

    def __call__(self, input, target):
        # do something to both images
        input = cv2.resize(input, self.size, interpolation=cv2.INTER_LINEAR)
        target = cv2.resize(target,self.size,interpolation=cv2.INTER_NEAREST)

        if self.rescale:
            input = (input/255.).astype(np.float32)
            # target = target/255.
            target = target.astype(np.float32)

        if(self.augment):
            # Random hflip
            hflip = random.random()
            if (hflip < 0.5):
                input = cv2.flip(input,1)
                target = cv2.flip(target, 1)

            #Random translation 0-2 pixels (fill rest with padding
            # transX = random.randint(-2, 2)
            # transY = random.randint(-2, 2)
            #
            # input = ImageOps.expand(input, border=(transX,transY,0,0), fill=0)
            # target = ImageOps.expand(target, border=(transX,transY,0,0), fill=255) #pad label filling with 255
            # input = input.crop((0, 0, input.size[0]-transX, input.size[1]-transY))
            # target = target.crop((0, 0, target.size[0]-transX, target.size[1]-transY))
        input = ToTensor()(input)
        target = ToLabel()(target)
        target = Relabel(255, 3)(target)

        return input, target


class CrossEntropyLoss2d(torch.nn.Module):

    def __init__(self, weight=None):
        super().__init__()

        self.loss = torch.nn.NLLLoss(weight)

    def forward(self, outputs, targets):
        return self.loss(torch.nn.functional.log_softmax(outputs, dim=1), targets)


def train(model, loader_train,loader_val, args, enc=False):
    best_acc = 0

    #TODO: calculate weights by processing dataset histogram (now its being set by hand from the torch values)
    #create a loder to run all images and calculate histogram of labels, then create weight array using class balancing

    # 这是给每一个类别，根据其出现的频率给一个对应的权重 TODO
    '''
    weight = torch.ones(NUM_CLASSES)
    weight[0] = 2.8149201869965	
    weight[1] = 6.9850029945374	
    weight[2] = 3.7890393733978	
    weight[3] = 9.9428062438965	
    weight[4] = 9.7702074050903	
    weight[5] = 9.5110931396484	
    weight[6] = 10.311357498169	
    weight[7] = 10.026463508606	
    weight[8] = 4.6323022842407	
    weight[9] = 9.5608062744141	
    weight[10] = 7.8698215484619	
    weight[11] = 9.5168733596802	
    weight[12] = 10.373730659485	
    weight[13] = 6.6616044044495	
    weight[14] = 10.260489463806	
    weight[15] = 10.287888526917	
    weight[16] = 10.289801597595	
    weight[17] = 10.405355453491	
    weight[18] = 10.138095855713	
    weight[19] = 0
    '''
    weight = torch.ones(NUM_CLASSES)
    weight[0] = 9.799923181533899
    weight[1] = 47.2
    weight[2] = 7.8698215484619
    weight[3] = 0

    if args.cuda:
        weight = weight.cuda()
    criterion = torch.nn.CrossEntropyLoss(weight)
    print(type(criterion))

    savedir = f'../save/{args.savedir}'

    automated_log_path = savedir + "/automated_log.txt"
    modeltxtpath = savedir + "/model.txt"

    if (not os.path.exists(automated_log_path)):    #dont add first line if it exists 
        with open(automated_log_path, "a") as myfile:
            myfile.write("Epoch\t\tTrain-loss\t\tTest-loss\t\tTrain-IoU\t\tTest-IoU\t\tlearningRate")

    with open(modeltxtpath, "w") as myfile:
        myfile.write(str(model))


    #TODO: reduce memory in first gpu: https://discuss.pytorch.org/t/multi-gpu-training-memory-usage-in-balance/4163/4        #https://github.com/pytorch/pytorch/issues/1893

    #optimizer = Adam(model.parameters(), 5e-4, (0.9, 0.999),  eps=1e-08, weight_decay=2e-4)     ## scheduler 1
    optimizer = Adam(model.parameters(), 5e-4, (0.9, 0.999),  eps=1e-08, weight_decay=1e-4)      ## scheduler 2

    start_epoch = 1
    if args.resume:
        #Must load weights, optimizer, epoch and best value.
        filenameCheckpoint = savedir + '/checkpoint.pth.tar'

        assert os.path.exists(filenameCheckpoint), "Error: resume option was used but checkpoint was not found in folder"
        checkpoint = torch.load(filenameCheckpoint)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_acc = checkpoint['best_acc']
        print("=> Loaded checkpoint at epoch {})".format(checkpoint['epoch']))

    #scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5) # set up scheduler     ## scheduler 1
    lambda1 = lambda epoch: pow((1-((epoch-1)/args.num_epochs)),0.9)  ## scheduler 2
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)                             ## scheduler 2

    if args.visualize and args.steps_plot > 0:
        board = Dashboard(args.port)

    for epoch in range(start_epoch, args.num_epochs+1):
        print("----- TRAINING - EPOCH", epoch, "-----")

        scheduler.step(epoch)    ## scheduler 2

        epoch_loss = []
        time_train = []

        doIouTrain = args.iouTrain   
        doIouVal =  args.iouVal      

        if (doIouTrain):
            iouEvalTrain = iouEval(NUM_CLASSES)

        usedLr = 0
        for param_group in optimizer.param_groups:
            print("LEARNING RATE: ", param_group['lr'])
            usedLr = float(param_group['lr'])

        model.train()
        for step, (images, labels,_,_) in enumerate(loader_train):

            start_time = time.time()
            #print (labels.size())
            #print (np.unique(labels.numpy()))
            #print("labels: ", np.unique(labels[0].numpy()))
            #labels = torch.ones(4, 1, 512, 1024).long()
            if args.cuda:
                images = images.cuda()
                labels = labels.cuda()
            #print("image: ", images.size())
            #print("labels: ", labels.size())
            inputs = Variable(images)
            targets = Variable(labels)
            outputs = model(inputs)

            #print("output: ", outputs.size()) #TODO
            #print("targets", np.unique(targets[:, 0].cpu().data.numpy()))

            optimizer.zero_grad()
            loss = criterion(outputs, targets[:, 0])

            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.data)
            time_train.append(time.time() - start_time)

            if (doIouTrain): # 计算IOU
                #start_time_iou = time.time()
                iouEvalTrain.addBatch(outputs.max(1)[1].unsqueeze(1).data, targets.data)
                #print ("Time to add confusion matrix: ", time.time() - start_time_iou)      

            #print(outputs.size())
            if args.visualize and args.steps_plot > 0 and step % args.steps_plot == 0:
                start_time_plot = time.time()
                image = inputs[0].cpu().data
                #image[0] = image[0] * .229 + .485
                #image[1] = image[1] * .224 + .456
                #image[2] = image[2] * .225 + .406
                #print("output", np.unique(outputs[0].cpu().max(0)[1].data.numpy()))
                board.image(image, f'input (epoch: {epoch}, step: {step})')
                if isinstance(outputs, list):   #merge gpu tensors
                    board.image(color_transform(outputs[0][0].cpu().max(0)[1].data.unsqueeze(0)),
                    f'output (epoch: {epoch}, step: {step})')
                else:
                    board.image(color_transform(outputs[0].cpu().max(0)[1].data.unsqueeze(0)),
                    f'output (epoch: {epoch}, step: {step})')
                board.image(color_transform(targets[0].cpu().data),
                    f'target (epoch: {epoch}, step: {step})')
                print ("Time to paint images: ", time.time() - start_time_plot)
            if args.steps_loss > 0 and step % args.steps_loss == 0:
                average = sum(epoch_loss) / len(epoch_loss)
                print(f'loss: {average:0.4} (epoch: {epoch}, step: {step})', 
                        "// Avg time/img: %.4f s" % (sum(time_train) / len(time_train) / args.batch_size))

            
        average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)
        
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
            iouEvalVal = iouEval(NUM_CLASSES)

        for step, (images, labels,_,_) in enumerate(loader_val):
            start_time = time.time()
            if args.cuda:
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


            #Add batch to calculate TP, FP and FN for iou estimation
            if (doIouVal):
                #start_time_iou = time.time()
                iouEvalVal.addBatch(outputs.max(1)[1].unsqueeze(1).data, targets.data)
                #print ("Time to add confusion matrix: ", time.time() - start_time_iou)

            if args.visualize and args.steps_plot > 0 and step % args.steps_plot == 0:
                start_time_plot = time.time()
                image = inputs[0].cpu().data
                board.image(image, f'VAL input (epoch: {epoch}, step: {step})')
                if isinstance(outputs, list):   #merge gpu tensors
                    board.image(color_transform(outputs[0][0].cpu().max(0)[1].data.unsqueeze(0)),
                    f'VAL output (epoch: {epoch}, step: {step})')
                else:
                    board.image(color_transform(outputs[0].cpu().max(0)[1].data.unsqueeze(0)),
                    f'VAL output (epoch: {epoch}, step: {step})')
                board.image(color_transform(targets[0].cpu().data),
                    f'VAL target (epoch: {epoch}, step: {step})')
                print ("Time to paint images: ", time.time() - start_time_plot)
            if args.steps_loss > 0 and step % args.steps_loss == 0:
                average = sum(epoch_loss_val) / len(epoch_loss_val)
                print(f'VAL loss: {average:0.4} (epoch: {epoch}, step: {step})', 
                        "// Avg time/img: %.4f s" % (sum(time_val) / len(time_val) / args.batch_size))
                       

        average_epoch_loss_val = sum(epoch_loss_val) / len(epoch_loss_val)
        #scheduler.step(average_epoch_loss_val, epoch)  ## scheduler 1   # update lr if needed

        iouVal = 0
        if (doIouVal):
            iouVal, iou_classes = iouEvalVal.getIoU()
            iouStr = getColorEntry(iouVal)+'{:0.2f}'.format(iouVal*100) + '\033[0m'
            print ("EPOCH IoU on VAL set: ", iouStr, "%") 
           

        # remember best valIoU and save checkpoint
        if iouVal == 0:
            current_acc = -average_epoch_loss_val
        else:
            current_acc = iouVal 
        is_best = current_acc > best_acc
        best_acc = max(current_acc, best_acc)
        if enc:
            filenameCheckpoint = savedir + '/checkpoint_enc.pth.tar'
            filenameBest = savedir + '/model_best_enc.pth.tar'    
        else:
            filenameCheckpoint = savedir + '/checkpoint.pth.tar'
            filenameBest = savedir + '/model_best.pth.tar'
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': str(model),
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer' : optimizer.state_dict(),
        }, is_best, filenameCheckpoint, filenameBest)

        #SAVE MODEL AFTER EPOCH
        if (enc):
            filename = f'{savedir}/model_encoder-{epoch:03}.pth'
            filenamebest = f'{savedir}/model_encoder_best.pth'
        else:
            filename = f'{savedir}/model-{epoch:03}.pth'
            filenamebest = f'{savedir}/model_best.pth'
        if args.epochs_save > 0 and step > 0 and step % args.epochs_save == 0:
            torch.save(model.state_dict(), filename)
            print(f'save: {filename} (epoch: {epoch})')
        if (is_best):
            torch.save(model.state_dict(), filenamebest)
            print(f'save: {filenamebest} (epoch: {epoch})')
            if (not enc):
                with open(savedir + "/best.txt", "w") as myfile:
                    myfile.write("Best epoch is %d, with Val-IoU= %.4f" % (epoch, iouVal))   
            else:
                with open(savedir + "/best_encoder.txt", "w") as myfile:
                    myfile.write("Best epoch is %d, with Val-IoU= %.4f" % (epoch, iouVal))           

        #SAVE TO FILE A ROW WITH THE EPOCH RESULT (train loss, val loss, train IoU, val IoU)
        #Epoch		Train-loss		Test-loss	Train-IoU	Test-IoU		learningRate
        with open(automated_log_path, "a") as myfile:
            myfile.write("\n%d\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.8f" % (epoch, average_epoch_loss_train, average_epoch_loss_val, iouTrain, iouVal, usedLr ))

    filenamelast = f'{savedir}/weight_final.pth'
    torch.save(model.state_dict(), filenamelast)

    return model   #return model (convenience for encoder-decoder training)

def test(filenameSave, model, dataloader_test, args):
    for step, (images, labels, filename, filenameGt) in enumerate(dataloader_test):
        if (args.cuda):
            images = images.cuda()
            # labels = labels.cuda()

        inputs = Variable(images)
        # targets = Variable(labels)
        with torch.no_grad():
            outputs = model(inputs)

        label = outputs[0].max(0)[1].byte().cpu().data
        # label_cityscapes = cityscapes_trainIds2labelIds(label.unsqueeze(0))
        label_color = Colorize()(label.unsqueeze(0))


        # image_transform(label.byte()).save(filenameSave)

        # label_save = ToPILImage()(label_color)
        label_save = label_color.numpy()
        label_save = label_save.transpose(1, 2, 0)
        # label_save.save(filenameSave)
        images = images.cpu().numpy().squeeze(axis=0).transpose(1, 2, 0)
        images = (images*255).astype(np.uint8)

        for i in range(len(filename)):
            fileSave = '../eval/'+ args.savedir + "/" + filename[i].split("leftImg8bit/")[1]
            os.makedirs(os.path.dirname(fileSave), exist_ok=True)
            output = cv2.addWeighted(images, 0.4, label_save, 0.6, 0)
            cv2.imwrite(fileSave,output)

            # print(fileSave)
            # plt.figure(figsize=(10.4,10.4), dpi=10)
            # # plt.imshow(images)
            # plt.imshow(label_save,alpha=0.6,cmap='gray')
            # plt.axis('off')
            # # plt.show()
            # plt.savefig(fileSave,dpi=10)
            # plt.close()

def inference(model, args):
    image_folder = "/media/pandongwei/ExtremeSSD/work_relative/extract_img/2020.10.16_1/"
    video_save_path = "/home/pandongwei/work_repository/erfnet_pytorch/eval/"

    # parameters about saving video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_save_path + 'output_new.avi', fourcc, 10.0, (640, 480))

    cuda = True
    model.eval()

    paths = []
    for root, dirs, files in os.walk(image_folder, topdown=True):
        for file in files:
            image_path = os.path.join(image_folder, file)
            paths.append(image_path)
    paths.sort()
    font = cv2.FONT_HERSHEY_SIMPLEX

    angle_pre = 0
    for i, path in enumerate(paths):
        start_time = time.time()
        image = cv2.imread(path)
        image = (image / 255.).astype(np.float32)

        image = ToTensor()(image).unsqueeze(0)
        if (cuda):
            image = image.cuda()

        input = Variable(image)

        with torch.no_grad():
            output = model(input)

        label = output[0].max(0)[1].byte().cpu().data
        # label_cityscapes = cityscapes_trainIds2labelIds(label.unsqueeze(0))
        label_color = Colorize()(label.unsqueeze(0))

        label_save = label_color.numpy()
        label_save = label_save.transpose(1, 2, 0)
        # 加上路径规划
        label_save, angle = perception_to_angle(label_save, angle_pre)
        # 加一个滤波以防止角度突然跳变 TODO
        if abs(angle - angle_pre) > 10:
            angle = angle_pre

        angle_pre = angle
        # label_save.save(filenameSave)
        image = image.cpu().numpy().squeeze(axis=0).transpose(1, 2, 0)
        image = (image * 255).astype(np.uint8)
        output = cv2.addWeighted(image, 0.5, label_save, 0.5, 0)
        cv2.putText(output,str(round(angle,3)),(50,50),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,0),2)
        #output = np.hstack([label_save, image])
        out.write(output)

        print(i, "  time: %.2f s" % (time.time() - start_time))
    out.release()

def img_to_video():
    img_path = '/media/pandongwei/ExtremeSSD/work_relative/extract_img/2020.10.23_1/'
    video_path = '/media/pandongwei/ExtremeSSD/work_relative/'
    # parameters about saving video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_path + 'video.avi', fourcc, 10.0, (640, 480))
    paths_img = []
    for dir_1 in os.listdir(img_path):
        path = img_path + dir_1
        paths_img.append(path)
    paths_img.sort()
    i = 0
    for path in paths_img:
        i += 1
        if i < 40 or i > 1050:
            continue
        img = cv2.imread(path)
        out.write(img)
    out.release()


def save_checkpoint(state, is_best, filenameCheckpoint, filenameBest):
    torch.save(state, filenameCheckpoint)
    if is_best:
        print("Saving model as best")
        torch.save(state, filenameBest)

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
            # 画轨迹图
            cv2.line(map, path_point[i], path_point[i+1], (80,80,255), 3)
        cv2.line(map, path_point[0], (path_point[0][0], path_point[0][1]-240), (0,0,0), 2)
        # 角度更新的方式
        start, end = path_point[0], path_point[-1]
        #angle = math.atan2(start[0]-end[0],start[1]-end[1])

        point_average = tuple([sum_point[0] // len(path_point), sum_point[1] // len(path_point)])
        angle = math.atan2(start[0]-point_average[0], start[1]-point_average[1])*180/math.pi

        # 加上 momenton
        angle = momenton * angle_pre + (1-momenton) * angle
        # 转向角度可视化
        cv2.line(map, path_point[0],(path_point[0][0]-int(math.tan(angle*math.pi/180.)*100),path_point[0][1]-100),(0,0,0),3)
    return map, angle

def main(args):
    savedir = f'../save/{args.savedir}'

    if not os.path.exists(savedir):
        os.makedirs(savedir)

    with open(savedir + '/opts.txt', "w") as myfile:
        myfile.write(str(args))

    #Load Model
    assert os.path.exists(args.model + ".py"), "Error: model definition not found"
    model = ERFNet(NUM_CLASSES)
    copyfile(args.model + ".py", savedir + '/' + args.model + ".py")
    
    if args.cuda:
        model = torch.nn.DataParallel(model).cuda()

    assert os.path.exists(args.datadir), "Error: datadir (dataset directory) could not be loaded"

    # co_transform = MyCoTransform(augment=True, rescale=True, width=640, height=480)#1024)
    # co_transform_val = MyCoTransform(augment=False, rescale=True, width=640, height=480)#1024)
    # dataset_train = cityscapes_cv(args.datadir, co_transform, 'train')
    # dataset_val = cityscapes_cv(args.datadir, co_transform_val, 'val')
    # loader_train = DataLoader(dataset_train, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True)
    # loader_val = DataLoader(dataset_val, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)

    # co_transform = MyCoTransform(augment=True, rescale=True, width=640, height=480)#1024)
    # dataset_train = gardenscapes(args.datadir, co_transform)
    # loader_train = DataLoader(dataset_train, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True)
    # loader_val = loader_train


    if args.state:
        #if args.state is provided then load this state for training
        #Note: this only loads initialized weights. If you want to resume a training use "--resume" option!!
        def load_my_state_dict(model, state_dict):  #custom function to load model when not all dict keys are there
            own_state = model.state_dict()
            for name, param in state_dict.items():
                if name not in own_state:
                     continue
                own_state[name].copy_(param)
            return model

        #print(torch.load(args.state))
        model = load_my_state_dict(model, torch.load(args.state))

    #model = train(model, loader_train, loader_val, args)   #Train decoder
    print("========== TRAINING FINISHED ===========")

    print("========== START TESTING ==============")
    model_dir = "/home/pandongwei/work_repository/erfnet_pytorch/save/"+args.savedir+'/model_best.pth'
    #model_dir = "/home/pan/repository/erfnet_pytorch/save/geoMat_regression_2/checkpoint.pth.tar"
    def load_my_state_dict(model, state_dict):
        # state_dict = state_dict["state_dict"]
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                 continue
            own_state[name].copy_(param)
        return model
    model = load_my_state_dict(model, torch.load(model_dir))
    # filenameSave = "../eval/" + args.savedir + "/"
    # os.makedirs(os.path.dirname(filenameSave), exist_ok=True)
    # co_transform_test = MyCoTransform(augment=False, rescale=True, width=640, height=480)  # 1024)
    # dataset_test = cityscapes_cv(args.datadir, co_transform_test, 'test')
    # loader_test = DataLoader(dataset_test,num_workers=args.num_workers, batch_size=1, shuffle=False)
    #test(filenameSave, model, loader_test, args)
    inference(model,args)

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  ## todo
    parser = ArgumentParser()
    parser.add_argument('--cuda', action='store_true', default=True)  #NOTE: cpu-only has not been tested so you might have to change code if you deactivate this flag
    parser.add_argument('--model', default="erfnet")
    parser.add_argument('--state')

    parser.add_argument('--port', type=int, default=8097)
    #parser.add_argument('--datadir', default="/home/disk1/pandongwei/cityscape/leftImg8bit_trainvaltest/")
    #parser.add_argument('--datadir', default="/media/pandongwei/ExtremeSSD/work_relative/cityscape/leftImg8bit_trainvaltest/")
    #parser.add_argument('--datadir', default='/home/disk1/pandongwei/extract_img/')
    parser.add_argument('--datadir', default="/media/pandongwei/ExtremeSSD/work_relative/extract_img/2020.10.16_1/")
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--num-epochs', type=int, default=150) #150
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--steps-loss', type=int, default=50)
    parser.add_argument('--steps-plot', type=int, default=50)
    parser.add_argument('--epochs-save', type=int, default=0)    #You can use this value to save model every X epochs
    parser.add_argument('--savedir', default="cityscape_4class_1")
    parser.add_argument('--decoder', action='store_true',default=True)
    parser.add_argument('--pretrainedEncoder', default="../trained_models/erfnet_encoder_pretrained.pth.tar")
    parser.add_argument('--visualize', action='store_true')

    parser.add_argument('--iouTrain', action='store_true', default=False) #recommended: False (takes more time to train otherwise)
    parser.add_argument('--iouVal', action='store_true', default=True)  
    parser.add_argument('--resume', action='store_true')    #Use this flag to load last checkpoint for training  

    main(parser.parse_args())
