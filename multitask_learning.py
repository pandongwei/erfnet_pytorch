# training the multitask learning model
# August 2020
# Dongwei Pan
################################################33

import os
import random
import time
import torch
import numpy as np
import cv2
from argparse import ArgumentParser
from torch.optim import SGD, Adam, lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from train.dataset import multitask_geoMat
from train.transform import Colorize
from train.multi_models import Multi_models
from tensorboardX import SummaryWriter


mean_and_var = 2

color_transform = Colorize(mean_and_var)

#Augmentations - different function implemented to perform random augments on both image and target
class MyCoTransform(object):
    def __init__(self, augment=True, rescale=True, size=104):

        self.augment = augment
        self.size = (size,size)
        self.rescale = rescale

        pass
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
        mean, var = outputs[:,0,:,:], outputs[:,1,:,:]
        # print(mean)
        # print(var)
        # print(targets)
        # print(torch.pow(mean - targets,2))
        # print(2*torch.pow(var,2))
        loss_1 = torch.mean(torch.pow(mean - targets,2)/(2*torch.pow(var,2)))
        loss_2 = torch.mean(torch.log(var))
        #loss_3,_ = torch.max(torch.mean(-var*100), 0)
        # 尝试使得网络输出log(var**2)而不是var，从而改变loss的表达形式，让loss1不会太大nan
        # loss_1 = torch.mean(0.5*torch.exp(-var)*torch.pow(mean-targets,2))
        # loss_2 = torch.mean(0.5*var)

        # print("loss1: ",loss_1)
        # print('loss2: ',loss_2)
        return 0.5*(loss_1 + 0.01*loss_2)

# L1 loss + punlishment
class L1loss_punlishment(torch.nn.Module):
    def __init__(self):
        super(L1loss_punlishment,self).__init__()

    def forward(self, outputs, targets):
        mean, var = outputs[:,0,:,:], outputs[:,1,:,:]
        loss_1 = torch.mean(torch.abs(mean - targets))
        loss_3 = torch.mean(torch.max(torch.zeros(mean.shape).cuda(), mean**2 - mean))
        # print("loss1: ",loss_1)
        # print('loss2: ',loss_2)
        return 0.5*(loss_1 + 2*loss_3)
# modified nagetive Gaussian log-likelihood loss with punishment
class GaussianLogLikelihoodLoss_punlishment(torch.nn.Module):
    def __init__(self):
        super(GaussianLogLikelihoodLoss_punlishment,self).__init__()

    def forward(self, outputs, targets):
        mean, log_var2 = outputs[:,0,:,:], outputs[:,1,:,:]
        loss_1 = torch.mean(torch.pow(mean - targets,2)/(2*torch.exp(log_var2)))
        loss_2 = 0.5*torch.mean(log_var2)
        loss_3 = torch.mean(torch.max(torch.zeros(mean.shape).cuda(), mean**2 - mean))
        # print("loss1: ",loss_1)
        # print('loss2: ',loss_2)
        return loss_1 + loss_2 + 2*loss_3

class Multi_loss(torch.nn.Module):
    def __init__(self, log_vars):
        super(Multi_loss,self).__init__()
        self.log_vars = log_vars
        self.l2Loss = torch.nn.MSELoss()
        self.l1Loss = torch.nn.L1Loss()
        self.crossEntropyLoss = torch.nn.CrossEntropyLoss()

    def forward(self, outputs, targets):
        output_traversability, output_depth, output_class = outputs[0], outputs[1], outputs[2]
        target_traversability, target_depth, target_class = targets[0], targets[1], targets[2]

        loss_1_1 = self.l2Loss(output_traversability, output_traversability)*torch.exp(-self.log_vars[0])
        loss_1_2 = self.log_vars[0]
        loss_1_3 = torch.mean(torch.max(torch.zeros(target_traversability.shape).cuda(), output_traversability**2 - output_traversability))
        loss_1 = loss_1_1 + loss_1_2 + loss_1_3

        loss_2 = self.l1Loss(output_depth, target_depth)*torch.exp(-self.log_vars[1]) + self.log_vars[1]
        loss_3 = 0.5*self.crossEntropyLoss(output_class, target_class)*torch.exp(-self.log_vars[2]) + self.log_vars[2]
        # print("loss1: ",loss_1)
        # print('loss2: ',loss_2)
        # print('loss3: ',loss_3)
        return loss_1 + loss_2 + loss_3

def train(savedir, model ,dataloader_train,dataloader_eval,criterion,optimizer, args):
    min_loss = float('inf')

    # use tensorboard
    writer = SummaryWriter(log_dir=savedir)

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

        scheduler.step(epoch)    

        epoch_loss = []
        time_train = []


        usedLr = 0
        for param_group in optimizer.param_groups:
            print("LEARNING RATE: ", param_group['lr'])
            usedLr = float(param_group['lr'])

        model.train()
        for step, (images, label_traver, label_depth, label_class, _) in enumerate(dataloader_train):

            start_time = time.time()
            #print (labels.size())
            #print (np.unique(labels.numpy()))
            #print("labels: ", np.unique(labels[0].numpy()))
            #labels = torch.ones(4, 1, 512, 1024).long()
            if args.cuda:
                images = images.cuda()
                label_traver = label_traver.cuda()
                label_depth = label_depth.cuda()
                label_class = label_class.cuda()
            #print("image: ", images.size())
            #print("labels: ", labels.size())
            inputs = Variable(images)
            targets_traver = Variable(label_traver)
            targets_depth = Variable(label_depth)
            targets_class = Variable(label_class)
            outputs = model(inputs)

            # print("output: ", outputs.size()) #TODO
            # print("targets", np.unique(targets[:, 0].cpu().data.numpy()))

            optimizer.zero_grad()
            loss = criterion(outputs, [targets_traver[:, 0], targets_depth[:, 0], targets_class])

            loss.backward()
            optimizer.step()

            epoch_loss.append(loss)
            time_train.append(time.time() - start_time)

            if args.steps_loss > 0 and step % args.steps_loss == 0:
                average = sum(epoch_loss) / len(epoch_loss)
                print(f'loss: {average:0.4} (epoch: {epoch}, step: {step})', 
                        "// Avg time/img: %.4f s" % (sum(time_train) / len(time_train) / args.batch_size))

            
        average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)
        writer.add_scalar('train_loss', average_epoch_loss_train, epoch)

        #Validate on 500 val images after each epoch of training
        print("----- VALIDATING - EPOCH", epoch, "-----")
        model.eval()
        epoch_loss_val = []
        time_val = []

        for step, (images, label_traver, label_depth, label_class, _) in enumerate(dataloader_eval):
            start_time = time.time()
            if args.cuda:
                images = images.cuda()
                label_traver = label_traver.cuda()
                label_depth = label_depth.cuda()
                label_class = label_class.cuda()
            #print("image: ", images.size())
            #print("labels: ", labels.size())
            inputs = Variable(images)
            targets_traver = Variable(label_traver)
            targets_depth = Variable(label_depth)
            targets_class = Variable(label_class)
            with torch.no_grad():
                outputs = model(inputs)

                loss = criterion(outputs, [targets_traver[:, 0], targets_depth[:, 0], targets_class])
            epoch_loss_val.append(loss.data)
            time_val.append(time.time() - start_time)

            if args.steps_loss > 0 and step % args.steps_loss == 0:
                average = sum(epoch_loss_val) / len(epoch_loss_val)
                print(f'VAL loss: {average:0.4} (epoch: {epoch}, step: {step})', 
                        "// Avg time/img: %.4f s" % (sum(time_val) / len(time_val) / args.batch_size))

        average_epoch_loss_val = sum(epoch_loss_val) / len(epoch_loss_val)
        #scheduler.step(average_epoch_loss_val, epoch)  ## scheduler 1   # update lr if needed
        writer.add_scalar('eval_loss', average_epoch_loss_val, epoch)

        is_best = average_epoch_loss_val < min_loss
        min_loss = min(min_loss, average_epoch_loss_val)

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

        filename = f'{savedir}/model-{epoch:03}.pth'
        filenamebest = f'{savedir}/model_best.pth'
        if args.epochs_save > 0 and step > 0 and step % args.epochs_save == 0:
            torch.save(model.state_dict(), filename)
            print(f'save: {filename} (epoch: {epoch})')
        if (is_best):
            torch.save(model.state_dict(), filenamebest)
            print(f'save: {filenamebest} (epoch: {epoch})')

        #SAVE TO FILE A ROW WITH THE EPOCH RESULT (train loss, val loss, train IoU, val IoU)
        #Epoch		Train-loss		Test-loss	Train-IoU	Test-IoU		learningRate
        with open(automated_log_path, "a") as myfile:
            myfile.write("\n%d\t\t%.4f\t\t%.4f\t\t%.8f" % (epoch, average_epoch_loss_train, average_epoch_loss_val, usedLr))
    writer.close()
    torch.save(model.state_dict(), f'{savedir}/weight_final.pth')
    return(model)   #return model (convenience for encoder-decoder training)


def test(filenameSave, model, dataloader_test, args):
    for step, (images, label_traver, _, _, filename) in enumerate(dataloader_test):
        if args.cuda:
            images = images.cuda()
            label_traver = label_traver.cuda()
        # print("image: ", images.size())
        # print("labels: ", labels.size())
        inputs = Variable(images)
        targets_traver = Variable(label_traver)
        with torch.no_grad():
            outputs = model(inputs)

        # print(outputs.shape)
        label = outputs[:,0,:,:].cpu().data
        # print(label.shape)
        #label_cityscapes = cityscapes_trainIds2labelIds(label.unsqueeze(0))
        label_color = (label.unsqueeze(1))
        # print(label_color.shape)

        # label_save = ToPILImage()(label_color)
        label_save = label_color.numpy()
        label_save = label_save.transpose(0, 2, 3, 1)
        # label_save.save(filenameSave)

        images = images.cpu().numpy()
        images = images.transpose(0, 2,3,1)

        # print(images.shape)
        # print(label_save.shape)
        for i in range(len(filename)):
            fileSave = '../eval/'+ args.savedir + filename[i].split("material_dataset_v2")[1]
            # print(fileSave)
            os.makedirs(os.path.dirname(fileSave), exist_ok=True)
            min_pixel, max_pixel, _, _ = cv2.minMaxLoc(label_save[i])
            print(min_pixel,'   ', max_pixel)
            # output = cv2.normalize(label_save[i], None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            output = label_save[i]
            output = output*255
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

def main(args):
    savedir = f'../save/{args.savedir}'
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    with open(savedir + '/opts.txt', "w") as myfile:
        myfile.write(str(args))

    #Load Model
    model = Multi_models()
    if args.cuda:
        model = torch.nn.DataParallel(model).cuda()

    assert os.path.exists(args.datadir), "Error: datadir (dataset directory) could not be loaded"
    co_transform = MyCoTransform(augment=True, rescale=True, size=args.size)
    co_transform_val = MyCoTransform(augment=False, rescale=True, size=args.size)
    dataset_train = multitask_geoMat(args.datadir, co_transform, 'train')
    dataset_val = multitask_geoMat(args.datadir, co_transform_val, 'test')

    loader_train = DataLoader(dataset_train, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True)
    loader_val = DataLoader(dataset_val, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)

    savedir = f'../save/{args.savedir}'

    # 定义task dependent log_variance 来进行multi-task learning
    log_var_a = torch.zeros((1,), requires_grad=True)
    log_var_b = torch.zeros((1,), requires_grad=True)
    log_var_c = torch.zeros((1,),requires_grad=True)
    log_vars = [log_var_a, log_var_b, log_var_c]
    params = ([p for p in model.parameters()] + log_vars)
    #optimizer = Adam(model.parameters(), 5e-4, (0.9, 0.999),  eps=1e-08, weight_decay=2e-4)     ## scheduler 1
    optimizer = Adam(params, 5e-4, (0.9, 0.999),  eps=1e-08, weight_decay=1e-4)

    criterion = Multi_loss(log_vars=log_vars)

    model = train(savedir, model, loader_train,loader_val,criterion,optimizer,args)   #Train decoder
    print("========== TRAINING FINISHED ===========")

    print("========== START TESTING ==============")
    # model_dir = "/home/pan/repository/erfnet_pytorch/save/"+args.savedir+'/weight_final.pth'
    # #model_dir = "/home/pan/repository/erfnet_pytorch/save/geoMat_regression_2/checkpoint.pth.tar"
    # def load_my_state_dict(model, state_dict):
    #     # state_dict = state_dict["state_dict"]
    #     own_state = model.state_dict()
    #     for name, param in state_dict.items():
    #         if name not in own_state:
    #              continue
    #         own_state[name].copy_(param)
    #     return model
    # model = load_my_state_dict(model, torch.load(model_dir))
    filenameSave = "./eval/" + args.savedir
    os.makedirs(os.path.dirname(filenameSave), exist_ok=True)
    test(filenameSave, model, loader_val, args)



if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"  ## todo
    parser = ArgumentParser()
    parser.add_argument('--cuda', action='store_true', default=True)  #NOTE: cpu-only has not been tested so you might have to change code if you deactivate this flag
    parser.add_argument('--model', default="erfnet")
    parser.add_argument('--state')

    parser.add_argument('--port', type=int, default=8097)
    parser.add_argument('--datadir', default="/mrtstorage/users/pan/material_dataset_v2/")
    parser.add_argument('--size', type=int, default=104)
    parser.add_argument('--num-epochs', type=int, default=50)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--steps-loss', type=int, default=50)
    parser.add_argument('--steps-plot', type=int, default=50)
    parser.add_argument('--epochs-save', type=int, default=0)    #You can use this value to save model every X epochs
    parser.add_argument('--savedir', default="geoMat_regression_13")
    parser.add_argument('--decoder', action='store_true',default=True)
    parser.add_argument('--pretrainedEncoder', default="")
    parser.add_argument('--resume', action='store_true')    #Use this flag to load last checkpoint for training  

    main(parser.parse_args())
