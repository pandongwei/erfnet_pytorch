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
from train.dataset import geoMat
from train.transform import Colorize
from train.erfnet import Net, Net_regression

import importlib
from iouEval import iouEval, getColorEntry

from shutil import copyfile


mean_and_var = 2

color_transform = Colorize(mean_and_var)

#Augmentations - different function implemented to perform random augments on both image and target
class MyCoTransform(object):
    def __init__(self, augment=True, rescale=True, size=104):

        self.augment = augment
        self.size = (size,size)

        pass
    def __call__(self, input, target):
        # do something to both images
        # input = Resize(self.size, Image.BILINEAR)(input)
        # target = Resize(self.size, Image.NEAREST)(target)
        input = cv2.resize(input,self.size,interpolation=cv2.INTER_LINEAR)
        target = cv2.resize(target,self.size,interpolation=cv2.INTER_NEAREST)

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
            
            #Random translation 0-2 pixels (fill rest with padding
            # transX = random.randint(-2, 2)
            # transY = random.randint(-2, 2)

            # input = ImageOps.expand(input, border=(transX,transY,0,0), fill=0)
            # target = ImageOps.expand(target, border=(transX,transY,0,0), fill=255) #pad label filling with 255
            # input = input.crop((0, 0, input.size[0]-transX, input.size[1]-transY))
            # target = target.crop((0, 0, target.size[0]-transX, target.size[1]-transY))

        input = ToTensor()(input)
        target = torch.from_numpy(np.array(target)).unsqueeze(0)
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


def train(savedir, model,dataloader_train,dataloader_eval,criterion,optimizer, args, enc=False):
    min_loss = float('inf')

    # use tensorboard
    writer = SummaryWriter(log_dir=savedir)
    if (enc):
        automated_log_path = savedir + "/automated_log_encoder.txt"
        modeltxtpath = savedir + "/model_encoder.txt"
    else:
        automated_log_path = savedir + "/automated_log.txt"
        modeltxtpath = savedir + "/model.txt"    

    if (not os.path.exists(automated_log_path)):    #dont add first line if it exists 
        with open(automated_log_path, "a") as myfile:
            myfile.write("Epoch\t\tTrain-loss\t\tTest-loss\t\tTrain-IoU\t\tTest-IoU\t\tlearningRate")

    with open(modeltxtpath, "w") as myfile:
        myfile.write(str(model))


    start_epoch = 1
    if args.resume:
        #Must load weights, optimizer, epoch and best value. 
        if enc:
            filenameCheckpoint = savedir + '/checkpoint_enc.pth.tar'
        else:
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



    for epoch in range(start_epoch, args.num_epochs+1):
        print("----- TRAINING - EPOCH", epoch, "-----")

        scheduler.step(epoch)    ## scheduler 2

        epoch_loss = []
        time_train = []

        doIouTrain = args.iouTrain   
        doIouVal =  args.iouVal      

        if (doIouTrain):
            iouEvalTrain = iouEval(mean_and_var)

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
            if args.cuda:
                images = images.cuda()
                labels = labels.cuda()
            #print("image: ", images.size())
            #print("labels: ", labels.size())
            inputs = Variable(images)
            targets = Variable(labels)
            outputs = model(inputs, only_encode=enc)

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
            iouEvalVal = iouEval(mean_and_var)

        for step, (images, labels,_) in enumerate(dataloader_eval):
            start_time = time.time()
            if args.cuda:
                images = images.cuda()
                labels = labels.cuda()
            optimizer.zero_grad()
            inputs = Variable(images)
            targets = Variable(labels)
            with torch.no_grad():
                outputs = model(inputs, only_encode=enc)

                loss = criterion(outputs, targets[:, 0])
            epoch_loss_val.append(loss.data)
            time_val.append(time.time() - start_time)

            if args.steps_loss > 0 and step % args.steps_loss == 0:
                average = sum(epoch_loss_val) / len(epoch_loss_val)
                print(f'VAL loss: {average:0.4} (epoch: {epoch}, step: {step})', 
                        "// Avg time/img: %.4f s" % (sum(time_val) / len(time_val) / args.batch_size))

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
            'best_acc': min_loss,
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
    writer.close()
    return(model)   #return model (convenience for encoder-decoder training)


def test(model_geoMat, model_freiburgForest, dataloader_test, cfg):
    cuda = cfg['cuda']
    savedir = cfg['savedir']
    data_savedir = f'eval/{savedir}'
    for step, (images, labels, filename) in enumerate(dataloader_test):
        if (cuda):
            images = images.cuda()
            #labels = labels.cuda()
        inputs = Variable(images)
        #targets = Variable(labels)
        with torch.no_grad():
            outputs_1 = model_freiburgForest(inputs)
            outputs_2 = model_geoMat(inputs)

        # 对输出做处理，得到可以显示的图像格式
        outputs_2 = outputs_2[:,1,:,:].cpu().data.unsqueeze(1)
        outputs_2 = outputs_2.numpy().transpose(0, 2, 3, 1)

        images = images.cpu().numpy()
        images = images.transpose(0, 2,3,1)

        # print(images.shape)
        # print(label_save.shape)
        for i in range(len(filename)):
            fileSave = data_savedir + filename[i].split("freiburg_forest_annotated")[1]
            os.makedirs(os.path.dirname(fileSave), exist_ok=True)
            min_pixel, max_pixel, _, _ = cv2.minMaxLoc(outputs_2[i])
            print(min_pixel,'   ', max_pixel)
            output = cv2.normalize(outputs_2[i], None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            output = output*255
            min_pixel, max_pixel, _, _ = cv2.minMaxLoc(output)
            print(min_pixel,'   ', max_pixel)
            cv2.imwrite(fileSave,output)

            # print(fileSave)
            # plt.figure(figsize=(10.4,10.4), dpi=10)
            # # plt.imshow(images)
            # plt.imshow(label_save,alpha=0.6,cmap='gray')
            # plt.axis('off')
            # # plt.show()
            # plt.savefig(fileSave,dpi=10)
            # plt.close()



def save_checkpoint(state, is_best, filenameCheckpoint, filenameBest):
    torch.save(state, filenameCheckpoint)
    if is_best:
        print("Saving model as best")
        torch.save(state, filenameBest)

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"  ## TODO
    # load parameters
    with open('config/config.json', 'r') as f:
        cfg = json.load(f)
    batch_size = int(cfg['batch_size'])
    cuda = cfg['cuda']
    model = cfg['model']
    size_geoMat = int(cfg['size_geoMat'])
    size_freiburgForest = int(cfg['size_freiburgForest'])
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

    data_savedir = f'eval/{savedir}'
    model_savedir = f'save/{savedir}'
    if not os.path.exists(data_savedir):
        os.makedirs(data_savedir)
    if not os.path.exists(model_savedir):
        os.makedirs(model_savedir)

    #Load both Models
    model_geoMat = Net_regression()
    model_freiburgForest = Net(7)
    # copyfile(args.model + ".py", savedir + '/' + args.model + ".py") # 在训练时候可以加上
    
    if cuda:
        model_geoMat = torch.nn.DataParallel(model_geoMat).cuda()
        model_freiburgForest = torch.nn.DataParallel(model_freiburgForest).cuda()

    assert os.path.exists(datadir_geoMat) or os.path.exists(datadir_freiburgForest), "Error: datadir (dataset directory) could not be loaded"

    # co_transform = MyCoTransform(augment=True, rescale=True, size=size)
    co_transform_val = MyCoTransform(augment=False, rescale=True, size=size_freiburgForest)
    # dataset_train = geoMat(datadir_geoMat, co_transform, 'train')
    dataset_val = geoMat(datadir_freiburgForest, co_transform_val, 'test')

    #loader_train = DataLoader(dataset_train, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True)
    loader_val = DataLoader(dataset_val, num_workers=num_workers, batch_size=batch_size, shuffle=False)

    criterion = GaussianLogLikelihoodLoss()
    #optimizer = Adam(model.parameters(), 5e-4, (0.9, 0.999),  eps=1e-08, weight_decay=2e-4)     ## scheduler 1
    optimizer = Adam(model.parameters(), 5e-4, (0.9, 0.999),  eps=1e-08, weight_decay=1e-4)

    #model = train(savedir, model, loader_train,loader_val,criterion,optimizer,False)   #Train decoder
    #print("========== TRAINING FINISHED ===========")

    print("========== START TESTING ==============")
    def load_my_state_dict(model, state_dict):
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                 continue
            own_state[name].copy_(param)
        return model
    model_geoMat = load_my_state_dict(model_geoMat, torch.load(weight_geoMat))
    model_freiburgForest = load_my_state_dict(model_freiburgForest, torch.load(weight_freigburgForest))

    test(model_geoMat, model_freiburgForest, loader_val, cfg)



if __name__ == '__main__':
    main()
