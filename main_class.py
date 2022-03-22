import os
import csv
import glob
import numpy as np
import torch
import time
import copy
import torch.nn as nn
from torch.optim import lr_scheduler
from utils import getFileList, Human_Hip_Joint_Score
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from models import resnet18,MobileNetV2,densenet121,vgg16,vgg16_bn
from PIL import Image
import random
import h5py
from scipy.io import loadmat
import argparse
import sys
parser = argparse.ArgumentParser(description='Classification.')
parser.add_argument('--data_dir', type=str, default='data/',
                    help='data directory')
parser.add_argument('--label_dir', type=str, default='Labels_v1.csv',
                    help='label directory')  
parser.add_argument('--num_class', type=int, default=2,
                    help='number of classes')                  
parser.add_argument('--feats_dir', type=str, default='feats/feat.h5',
                    help='feat directory')                                     
parser.add_argument('--img_type', type=str, default='intensity',
                    help='type of image')
parser.add_argument('--img_wid', type=int, default=480,
                    help='image width')
parser.add_argument('--img_hei', type=int, default=360,
                    help='image width')    
parser.add_argument('--scaling_fac', type=int, default=2,
                    help='scaling factor')    
parser.add_argument('--train_ratio', type=float, default=0.8,
                    help='ratio of training samples')   
parser.add_argument('--val_ratio', type=float, default=0.1,
                    help='ratio of validation samples') 
parser.add_argument('--rndSeed', type=int, default=2021, #0 
                    help='rnd seed') 
parser.add_argument('--rndSeed2', type=int, default=20214,
                    help='rnd seed')                     
parser.add_argument('--num_worker', type=int, default=16,
                    help='number of workers')    
parser.add_argument('--model_select', type=str, default='vgg16',
                    help='model to be applied')   
parser.add_argument('--criterion', type=str, default='CE',
                    help='criterion to be applied')                                                                                                                                                
parser.add_argument('--epoch', type=int, default=150,
                    help='number of epochs')
parser.add_argument('--batch_size', type=int, default=128,
                    help='batch size')                    
parser.add_argument('--lr', type=float, default=0.0005, #0.0001
                    help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0005,
                    help='weight decay')
parser.add_argument('--step_size', type=int, default=7, 
                    help='step size')
parser.add_argument('--gamma', type=float, default=0.2, 
                    help='gamma value')
parser.add_argument('--ckptFile', default='checkPoint_CRI_CL', ## CRI for Criterions
                    help='checkpoint save root')
parser.add_argument('--results', default='results_CRI_CL_',
                    help='path to save recorded results')                    
parser.add_argument('--gpu_id', default='0',
                    help='GPU ID')                    
args = parser.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_id
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

ckptFile_path = args.ckptFile + str(args.num_class)
results_path = args.results + str(args.num_class)
os.makedirs(ckptFile_path, exist_ok=True)
os.makedirs(results_path, exist_ok=True)
os.makedirs('logs', exist_ok=True)
file_name_base = '%s_%s_train_val_%s_%s_epoch_%d_seed_%d'%(args.model_select,\
                   args.img_type, str(args.train_ratio), str(args.val_ratio), args.epoch, args.rndSeed)
ckpt_path_save = os.path.join(ckptFile_path, file_name_base + ".pth")
ckpt_path_save_last = os.path.join(ckptFile_path, file_name_base + "_last.pth")
log_path_save = os.path.join(results_path, file_name_base + ".out")
sys.stdout=open(log_path_save,"w")
# print(args)


if args.img_type == 'multi':
    num_chan = 2
    feats = []
elif args.img_type == 'feat_multi':  
    num_chan = 19
    feats = h5py.File(args.feats_dir, 'r')
elif args.img_type == 'feat':
    num_chan = 17 
    feats = h5py.File(args.feats_dir, 'r')     
else: 
    num_chan = 1 
    feats = []

if args.model_select == 'vgg16':
    # model = vgg16(num_chan=num_chan, num_classes=args.num_class).to(device) 
    model = vgg16_bn(num_chan=num_chan, num_classes=args.num_class).to(device)
elif args.model_select == 'resnet18':
    model = resnet18(num_chan=num_chan, num_classes=args.num_class).to(device)  
elif args.model_select == 'densenet121':
    model = densenet121(num_chan=num_chan, num_classes=args.num_class).to(device)
elif args.model_select == 'mobilenetv2':
    model = MobileNetV2(num_chan=num_chan, num_classes=args.num_class).to(device)  
print(model)   

if args.criterion == 'CE': 
    criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, \
                             weight_decay=args.weight_decay)
scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, \
                                       gamma=args.gamma)

# import torch.optim as optim
# optimizer = optim.SGD(model.parameters(), lr=0.01, #0.1,1e-4 not work;1e-2 works;
#                       momentum=0.9, weight_decay=5e-4)               
# # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
# scheduler = lr_scheduler.StepLR(optimizer, step_size=5, \
#                                        gamma=args.gamma)

transform_data_train = transforms.Compose([                                      
                  transforms.Resize((args.img_wid // args.scaling_fac,
                  args.img_hei //args.scaling_fac)),
                  transforms.RandomHorizontalFlip(p=0.5),
                #   transforms.RandomRotation(degrees=(0, 30)),
                  transforms.ToTensor()
                  ])
transform_data_test = transforms.Compose([                                      
                  transforms.Resize((args.img_wid // args.scaling_fac,
                  args.img_hei //args.scaling_fac)),
                  transforms.ToTensor()
                  ])        

def getFileList_folders(root_dir, folders_dir, img_type='intensity'):
    if img_type == 'multi' or img_type == 'feat_multi' or img_type == 'feat':
        img_type = 'intensity'
    folders_select = [os.path.join(root_dir, folder, img_type) for folder in folders_dir]
    files_path = []
    for folder in folders_select:
        files_list = glob.glob(os.path.join(folder, '*.png')) + glob.glob(os.path.join(folder, '*.jpg'))
        for file_list in files_list:
            files_path.append(file_list)      
    return files_path

# def get_labels(imgs_path, folders_Label):
#     imgs_label = [] 
#     label_min = 0
#     label_max = np.array(folders_Label).max()
#     class_interval = np.ceil((label_max - label_min) / args.num_class)  
#     num_cls1 = 0 
#     num_cls0 = 0
#     for img_path in imgs_path:
#         folder_name_tmp = img_path.split('/')[1]
#         folder_name = folder_name_tmp.split('_')[1]
#         label_tmp = folders_dict[folder_name]
#         label = int(np.ceil(label_tmp // class_interval ) )
#         if label == 1:
#             num_cls1 += 1
#         else:
#             num_cls0 += 1
#         # print(label_tmp, label, num_cls0, num_cls1)
#         imgs_label.append(label)
#     return imgs_label


def get_labels_CL2(imgs_path, folders_Label):
    imgs_label = [] 
    label_mid_idx = int(np.floor(len(folders_Label)/args.num_class)) - 1
    folders_Label_tmp = sorted(folders_Label)
    label_mid = 5 ## folders_Label_tmp[label_mid_idx]. Use 5 as interval
    # print(label_mid_idx)
    # print(label_mid) ##6.685185185; 11880 vs 11419
    cl0, cl1 = 0, 0

    for img_path in imgs_path:
        folder_name_tmp = img_path.split('/')[1]
        folder_name = folder_name_tmp.split('_')[1]
        label_tmp = folders_dict[folder_name]

        if label_tmp <= label_mid:
            label = 0
            cl0 += 1
        else:
            label = 1
            cl1 += 1
        imgs_label.append(label)
        # print(img_path, label_tmp, label, cl0, cl1)

    return imgs_label


def get_labels_CL3(imgs_path, folders_Label):
    imgs_label = [] 
    label_mid_idx = int(np.floor(len(folders_Label)/args.num_class)) - 1 
    folders_Label_tmp = sorted(folders_Label)
    label_int1 = 5 ## folders_Label_tmp[label_mid_idx] Use 5 as 1st interval 
    label_int2 = 10 ##folders_Label_tmp[label_mid_idx*2] Use 10 as 2nd interval
    # print(label_int1, label_int2) ##5.555555556 10.3993
    cl0, cl1, cl2 = 0, 0, 0

    for img_path in imgs_path:
        folder_name_tmp = img_path.split('/')[1]
        folder_name = folder_name_tmp.split('_')[1]
        label_tmp = folders_dict[folder_name]

        if label_tmp <= label_int1:
            label = 0
            cl0 += 1
        elif label_tmp <= label_int2:
            label = 1
            cl1 += 1
        else:
            label = 2
            cl2 += 1
        imgs_label.append(label)
        # print(img_path, label_tmp, label, cl0, cl1, cl2)
    return imgs_label

# labels
imgs_label = []
folders_name = []
folders_Label = []
with open(args.label_dir, mode='r') as infile:
    reader = csv.DictReader(infile)
    for row in reader:
        curr_folder = row['folderID']
        if curr_folder == '026-S-1.1' or curr_folder == '016-SA-1.1':
            pass
        else:
            folders_name.append(row['folderID'])
            folders_Label.append(float(row['folderLabel']))
    
folders_dict = dict(zip(folders_name, folders_Label))

folders = os.listdir(args.data_dir)
num_train_folder, num_val_folder = int(np.ceil(args.train_ratio * len(folders))), \
                        int(np.ceil(args.val_ratio * len(folders)))
num_test_folder = len(folders) - num_train_folder - num_val_folder
# print(len(folders_Label), num_train_folder, num_val_folder, num_test_folder) #94, 12, 11

# folders
random.seed(args.rndSeed)
seq_folder = list(range(len(folders)))
random.shuffle(seq_folder)

train_indx_folder, val_indx_folder = seq_folder[:num_train_folder], seq_folder[num_train_folder:num_train_folder+num_val_folder]
test_indx_folder = seq_folder[num_train_folder + num_val_folder:]

train_folders = [folders[idx] for idx in train_indx_folder] 
val_folders = [folders[idx] for idx in val_indx_folder]
test_folders = [folders[idx] for idx in test_indx_folder]


## img lists
imgs_train = getFileList_folders(args.data_dir, train_folders, img_type=args.img_type)
imgs_val = getFileList_folders(args.data_dir, val_folders, img_type=args.img_type)
imgs_test = getFileList_folders(args.data_dir, test_folders, img_type=args.img_type)
print('\ndata split: train/val/test=%d/%d/%d.\n' %(len(imgs_train),len(imgs_val),len(imgs_test)))  # train/val/test=23299/3008/2760

if args.num_class == 2:
    labels_train = get_labels_CL2(imgs_train, folders_Label=folders_Label) ## interval 5: 6120 17193
    labels_val = get_labels_CL2(imgs_val, folders_Label=folders_Label) ## interval 5: 510 2535
    labels_test = get_labels_CL2(imgs_test, folders_Label=folders_Label) ## interval 5: 255 2454
elif args.num_class == 3:
    labels_train = get_labels_CL3(imgs_train, folders_Label=folders_Label)  ## interval 5/10: 6120 9506 7687
    labels_val = get_labels_CL3(imgs_val, folders_Label=folders_Label)      ## interval 5/10: 510 240 2295
    labels_test = get_labels_CL3(imgs_test, folders_Label=folders_Label)    ## interval 5/10: 255 765 1689
else:
    raise ValueError("Only support two/three classes case!")

## shuffle training data
random.seed(args.rndSeed2)
seq = list(range(len(imgs_train)))
random.shuffle(seq)

imgs_train = [imgs_train[idx] for idx in seq]
labels_train = [labels_train[idx] for idx in seq]

# import matplotlib as mpl
# mpl.use('Agg')
# fig = plt.figure()
# plt.hist(folders_Label, density=False, bins=50)  # density=False would make counts
# plt.ylabel('Count')
# plt.xlabel('Label')
# fig.savefig('label_hist.png')
# plt.show()


## create dataset
score_dataset_train = Human_Hip_Joint_Score(imgs_path=imgs_train, feats_dir=args.feats_dir, \
                                    imgs_label=labels_train, transform=transform_data_train, img_type=args.img_type) 
score_dataset_val = Human_Hip_Joint_Score(imgs_path=imgs_val, feats_dir=args.feats_dir, \
                                    imgs_label=labels_val, transform=transform_data_test, img_type=args.img_type)
score_dataset_test = Human_Hip_Joint_Score(imgs_path=imgs_test, feats_dir=args.feats_dir, \
                                    imgs_label=labels_test, transform=transform_data_test, img_type=args.img_type)
train_loader = DataLoader(score_dataset_train, batch_size=args.batch_size, shuffle=True,num_workers=args.num_worker)
val_loader = DataLoader(score_dataset_val, batch_size=args.batch_size, shuffle=True,num_workers=args.num_worker)
test_loader = DataLoader(score_dataset_test, batch_size=args.batch_size, shuffle=True,num_workers=args.num_worker)
dataloaders = {'train': train_loader, 'val':val_loader, 'test':test_loader}
dataset_sizes = {'train': len(train_loader), 'val': len(val_loader), 'test':len(test_loader)}
#print(dataset_sizes['test'])  #'train':83; 'val':24 ; 'test':22; num of epochs
#a = next(iter(dataloaders['train']))
#print(a['image'].shape)

import sklearn.metrics as skmet

def get_acc(lab_real, lab_pred, verbose=False):
    acc = skmet.accuracy_score(lab_real, lab_pred)
    return acc

def get_precision(lab_real, lab_pred, verbose=False):
    precision = skmet.precision_score(lab_real, lab_pred, average='weighted')
    return precision

def get_recall(lab_real, lab_pred, verbose=False):
    recall = skmet.recall_score(lab_real, lab_pred, average='weighted')
    return recall

def get_f1(lab_real, lab_pred, verbose=False):
    f1 = skmet.f1_score(lab_real, lab_pred, average='weighted')
    return f1

def get_confusion(lab_real, lab_pred, verbose=False):
    confusion_matrix = skmet.confusion_matrix(lab_real, lab_pred)
    return confusion_matrix


EPS = 1e-10

def train_model(model, criterion, optimizer, scheduler, num_epochs=150):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 100)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  
            else:
                model.eval()   

            running_loss = 0
            running_acc = 0
            running_prec = 0
            running_recall = 0
            running_f1 = 0
            tps = [0] * args.num_class

            for batch_data in dataloaders[phase]:
                inputs, labels = batch_data['image'], batch_data['label']
                inputs = inputs.to(device)
                labels = labels.long().to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() 
                running_acc += get_acc(labels.data.cpu().numpy(), preds.data.cpu().numpy())
                running_prec += get_precision(labels.data.cpu().numpy(), preds.data.cpu().numpy())
                running_recall += get_recall(labels.data.cpu().numpy(), preds.data.cpu().numpy())
                running_f1 += get_f1(labels.data.cpu().numpy(), preds.data.cpu().numpy())
                
                # matrix_tmp = get_confusion(labels.data.cpu().numpy(), preds.data.cpu().numpy())
                # matrix = matrix_tmp.diagonal()/(matrix_tmp.sum(axis=1) + EPS )
                # tps = [tps[k] + matrix[k] for k in range(args.num_class)]

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_acc / dataset_sizes[phase]
            epoch_prec = running_prec / dataset_sizes[phase]
            epoch_recall = running_recall / dataset_sizes[phase]
            epoch_f1 = running_f1 / dataset_sizes[phase]
            # epoch_tps = [tps[k] / dataset_sizes[phase] for k in range(args.num_class)]

            # if args.num_class == 2:
            #     print('{} Loss: {:.4f}, acc: {:.4f}, prec: {:.4f}, recall: {:.4f}, f1: {:.4f}, tpr-cl1: {:.4f}, tpr-cl2: {:.4f}'.format(phase, epoch_loss, epoch_acc, epoch_prec, \
            #             epoch_recall, epoch_f1, epoch_tps[0], epoch_tps[1]))
            # elif args.num_class == 3:
            #     print('{} Loss: {:.4f}, acc: {:.4f}, prec: {:.4f}, recall: {:.4f}, f1: {:.4f}, tpr-cl1: {:.4f}, tpr-cl2: {:.4f}, tpr-cl3: {:.4f}'.format(phase, epoch_loss, \
            #             epoch_acc, epoch_prec, epoch_recall, epoch_f1, \
            #             epoch_tps[0], epoch_tps[1], epoch_tps[2]))     
            if args.num_class == 2:
                print('{} Loss: {:.4f}, acc: {:.4f}, prec: {:.4f}, recall: {:.4f}, f1: {:.4f}'.format(phase, epoch_loss, \
                    epoch_acc, epoch_prec, epoch_recall, epoch_f1))
            elif args.num_class == 3:
                print('{} Loss: {:.4f}, acc: {:.4f}, prec: {:.4f}, recall: {:.4f}, f1: {:.4f}'.format(phase, epoch_loss, \
                        epoch_acc, epoch_prec, epoch_recall, epoch_f1))           
            
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            
            if phase == 'val':
                test_model(model)
            
            scheduler.step()

        print()
        

    time_elapsed = time.time() - since

    print('-' * 100)
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val acc: {:.4f}'.format(best_acc))

    torch.save(model.state_dict(), ckpt_path_save_last )  # save last model

    model.load_state_dict(best_model_wts)
    return model


def test_model(pretrained_model, phase='test'):
    pretrained_model.eval()
    running_acc = 0
    running_prec = 0
    running_recall = 0
    running_f1 = 0
    tps = [0] * args.num_class

    for batch_data in dataloaders[phase]:
        inputs, labels = batch_data['image'], batch_data['label']
        inputs = inputs.to(device)
        labels = labels.long().to(device)
        with torch.no_grad():
            outputs = pretrained_model(inputs)
            _, preds = torch.max(outputs, 1)

        running_acc += get_acc(labels.data.cpu().numpy(), preds.data.cpu().numpy())
        running_prec += get_precision(labels.data.cpu().numpy(), preds.data.cpu().numpy())
        running_recall += get_recall(labels.data.cpu().numpy(), preds.data.cpu().numpy())
        running_f1 += get_f1(labels.data.cpu().numpy(), preds.data.cpu().numpy())

    acc = running_acc / dataset_sizes[phase]
    prec = running_prec / dataset_sizes[phase]
    recall = running_recall / dataset_sizes[phase]
    f1 = running_f1 / dataset_sizes[phase]

    if args.num_class == 2:
        print('Testing acc: {:.4f}, prec: {:.4f}, recall: {:.4f}, f1: {:.4f}'.format(acc, prec, \
                recall, f1))
    elif args.num_class == 3:
        print('Testing acc: {:.4f}, prec: {:.4f}, recall: {:.4f}, f1: {:.4f}'.format(acc, prec, \
                recall, f1))
    print('-' * 100)

def test_model_cls(pretrained_model, phase='test'):
    pretrained_model.eval()
    running_acc = 0
    running_prec = 0
    running_recall = 0
    running_f1 = 0
    tps = [0] * args.num_class

    for batch_data in dataloaders[phase]:
        inputs, labels = batch_data['image'], batch_data['label']
        inputs = inputs.to(device)
        labels = labels.long().to(device)
        with torch.no_grad():
            outputs = pretrained_model(inputs)
            _, preds = torch.max(outputs, 1)

        running_acc += get_acc(labels.data.cpu().numpy(), preds.data.cpu().numpy())
        running_prec += get_precision(labels.data.cpu().numpy(), preds.data.cpu().numpy())
        running_recall += get_recall(labels.data.cpu().numpy(), preds.data.cpu().numpy())
        running_f1 += get_f1(labels.data.cpu().numpy(), preds.data.cpu().numpy())

        matrix_tmp = get_confusion(labels.data.cpu().numpy(), preds.data.cpu().numpy())
        matrix = matrix_tmp.diagonal()/(matrix_tmp.sum(axis=1) + EPS )
        if len(matrix) != args.num_class:
            print(labels.data, preds.data)
            raise ValueError("tps dim not equal to cls!")
        else:
            tps = [tps[k] + matrix[k] for k in range(args.num_class)]

    acc = running_acc / dataset_sizes[phase]
    prec = running_prec / dataset_sizes[phase]
    recall = running_recall / dataset_sizes[phase]
    f1 = running_f1 / dataset_sizes[phase]

    tps = [tps[k] / dataset_sizes[phase] for k in range(args.num_class)]

    print('-' * 50)
    if args.num_class == 2:
        print('Testing acc: {:.4f}, prec: {:.4f}, recall: {:.4f}, f1: {:.4f}, tpr-cl1: {:.4f}, tpr-cl2: {:.4f}'.format(acc, prec, \
                recall, f1, tps[0], tps[1]))
    elif args.num_class == 3:
        print('Testing acc: {:.4f}, prec: {:.4f}, recall: {:.4f}, f1: {:.4f}, tpr-cl1: {:.4f}, tpr-cl2: {:.4f}, tpr-cl3: {:.4f}'.format(acc, prec, \
                recall, f1, tps[0], tps[1], tps[2])) 

    print('-' * 100)       

if not os.path.isfile(ckpt_path_save):
    print("\nBegin training CNN: ")
    model.train()
    model = train_model(model, criterion, optimizer, scheduler, 
                       num_epochs=args.epoch) 
    torch.save(model.state_dict(),ckpt_path_save)
    print("ckpt saved at: %s !\n" %(ckpt_path_save))

else:
    print("\nCkpt already exists, loading ...")
    model.load_state_dict(torch.load(ckpt_path_save))

## test 
model.eval()
test_model_cls(model)
torch.cuda.empty_cache() 
sys.stdout.close()

if args.img_type == 'feat_multi' or args.img_type == 'feat':
    feats.close()


