# Code to produce colored segmentation output in Pytorch for all cityscapes subsets  
# Sept 2017
# Eduardo Romera
#######################

import numpy as np
import torch
import os
import importlib
import matplotlib.pyplot as plt
from PIL import Image
from argparse import ArgumentParser

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize
from torchvision.transforms import ToTensor, ToPILImage

from dataset import freiburgForest
from erfnet import ERFNet
from transform import Relabel, ToLabel, Colorize

import visdom


NUM_CHANNELS = 3
NUM_CLASSES = 7

image_transform = ToPILImage()
input_transform_cityscapes = Compose([
    Resize((512,1024),Image.BILINEAR),
    ToTensor(),
    #Normalize([.485, .456, .406], [.229, .224, .225]),
])
target_transform_cityscapes = Compose([
    Resize((512,1024),Image.NEAREST),
    ToLabel(),
    Relabel(0, 1),
    Relabel(35, 2),
    Relabel(96, 2),
    Relabel(100, 4),
    Relabel(150, 3),
    Relabel(170, 6),
    Relabel(255, 5),
])

def main(args):

    modelpath = args.loadDir + args.loadModel
    #weightspath = args.loadDir + args.loadWeights #TODO
    weightspath = "../save/feriburgForest_1/model_best_1.pth"
    print ("Loading model: " + modelpath)
    print ("Loading weights: " + weightspath)

    #Import ERFNet model from the folder
    #Net = importlib.import_module(modelpath.replace("/", "."), "ERFNet")
    model = ERFNet(NUM_CLASSES)
  
    model = torch.nn.DataParallel(model)
    if (not args.cpu):
        model = model.cuda()

    #model.load_state_dict(torch.load(args.state))
    #model.load_state_dict(torch.load(weightspath)) #not working if missing key

    def load_my_state_dict(model, state_dict):  #custom function to load model when not all dict elements
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                 continue
            own_state[name].copy_(param)
        return model

    model = load_my_state_dict(model, torch.load(weightspath))
    print("Model and weights LOADED successfully")

    model.eval()

    if(not os.path.exists(args.datadir)):
        print ("Error: datadir could not be loaded")


    loader = DataLoader(freiburgForest(args.datadir, input_transform_cityscapes, target_transform_cityscapes, subset = args.subset),
        num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)

    # For visualizer:
    # must launch in other window "python3.6 -m visdom.server -port 8097"
    # and access localhost:8097 to see it
    if (args.visualize):
        vis = visdom.Visdom()

    for step, (images, labels, filename, filenameGt) in enumerate(loader):
        if (not args.cpu):
            images = images.cuda()
            #labels = labels.cuda()

        inputs = Variable(images)
        #targets = Variable(labels)
        with torch.no_grad():
            outputs = model(inputs)

        label = outputs[0].max(0)[1].byte().cpu().data
        #label_cityscapes = cityscapes_trainIds2labelIds(label.unsqueeze(0))
        label_color = Colorize()(label.unsqueeze(0))

        filenameSave = "./freiburgforest_1/" + filename[0].split("freiburg_forest_annotated/")[1]
        os.makedirs(os.path.dirname(filenameSave), exist_ok=True)
        #image_transform(label.byte()).save(filenameSave)

        # label_save = ToPILImage()(label_color)
        label_save = label_color.numpy()
        label_save = label_save.transpose(1, 2, 0)
        # label_save.save(filenameSave)
        images = images.cpu().numpy().squeeze(axis=0)
        images = images.transpose(1,2,0)

        # print(images.shape)
        # print(label_save.shape)
        plt.figure(figsize=(10.24, 5.12), dpi=100)
        plt.imshow(images)
        plt.imshow(label_save,alpha=0.6)
        plt.axis('off')
        # plt.show()
        plt.savefig(filenameSave,dpi=100)
        plt.close()

        if (args.visualize):
            vis.image(label_color.numpy())
        print (step, filenameSave)

    

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  ## todo
    parser = ArgumentParser()

    parser.add_argument('--state')

    parser.add_argument('--loadDir',default="../trained_models/")
    parser.add_argument('--loadWeights', default="erfnet_pretrained.pth")
    parser.add_argument('--loadModel', default="erfnet.py")
    parser.add_argument('--subset', default="test")  #can be val, test, train, demoSequence

    parser.add_argument('--datadir', default="/mrtstorage/users/pan/freiburg_forest_multispectral_annotated/freiburg_forest_annotated/")
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')

    parser.add_argument('--visualize', action='store_true')
    main(parser.parse_args())
