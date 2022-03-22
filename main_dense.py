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
from models import resnet18,MobileNetV2,densenet121,vgg16
from PIL import Image
import random
import h5py
from scipy.io import loadmat
import argparse
import sys
parser = argparse.ArgumentParser(description='score prediction.')
parser.add_argument('--data_dir', type=str, default='data/',
                    help='data directory')
parser.add_argument('--label_dir', type=str, default='Labels_dense.mat',
                    help='label directory')   
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
parser.add_argument('--rndSeed', type=int, default=2021,
                    help='rnd seed') 
parser.add_argument('--rndSeed2', type=int, default=20214,
                    help='rnd seed')                     
parser.add_argument('--num_worker', type=int, default=16,
                    help='number of workers')    
parser.add_argument('--model_select', type=str, default='resnet18',
                    help='model to be applied')   
parser.add_argument('--criterion', type=str, default='MSE',
                    help='criterion to be applied')                                                                                                                                                
parser.add_argument('--epoch', type=int, default=150,
                    help='number of epochs')
parser.add_argument('--batch_size', type=int, default=128,
                    help='batch size')                    
parser.add_argument('--lr', type=float, default=0.0005, #0.0001
                    help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0005,
                    help='weight decay')
parser.add_argument('--step_size', type=int, default=20, 
                    help='step size')
parser.add_argument('--gamma', type=float, default=0.2, 
                    help='gamma value')
parser.add_argument('--ckptFile', default='checkPoint_dense_col2',
                    help='checkpoint save root')
parser.add_argument('--results', default='results_dense_col2',
                    help='path to save recorded results')                    
parser.add_argument('--gpu_id', default='0',
                    help='GPU ID')                    
args = parser.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_id
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

os.makedirs(args.ckptFile, exist_ok=True)
os.makedirs(args.results, exist_ok=True)
os.makedirs('logs', exist_ok=True)
file_name_base = '%s_%s_train_val_%s_%s_epoch_%d_seed_%d'%(args.model_select,\
                   args.img_type, str(args.train_ratio), str(args.val_ratio), args.epoch, args.rndSeed)
ckpt_path_save = os.path.join(args.ckptFile, file_name_base + ".pth")
log_path_save = os.path.join(args.results, file_name_base + ".out")
sys.stdout=open(log_path_save,"w")
print(args)


if args.img_type == 'multi':
    num_chan = 2
    #feats = []
elif args.img_type == 'feat_multi':  
    num_chan = 19
    #feats = h5py.File(args.feats_dir, 'r')
elif args.img_type == 'feat':
    num_chan = 17 
    #feats = h5py.File(args.feats_dir, 'r')     
else: 
    num_chan = 1 
    #feats = []

if args.model_select == 'vgg16':
    model = vgg16(num_chan=num_chan).to(device) 
elif args.model_select == 'resnet18':
    model = resnet18(num_chan=num_chan).to(device)  
elif args.model_select == 'densenet121':
    model = densenet121(num_chan=num_chan).to(device)
elif args.model_select == 'mobilenetv2':
    model = MobileNetV2(num_chan=num_chan).to(device)    

if args.criterion == 'MSE': 
    criterion = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, \
                             weight_decay=args.weight_decay)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, \
                                       gamma=args.gamma)

transform_data = transforms.Compose([                                      
                  transforms.Resize((args.img_wid // args.scaling_fac,
                  args.img_hei //args.scaling_fac)),
                  transforms.ToTensor()
                  ])


def read_labels_from_mat(mat_file_name):
    """Read dense labels from .mat file.
    Input: mat_file_name, i.e., path to mat file.
    Return: a dict with folder_name as keys, 
            and intensity/phase arrays as values 
            I.e., (value[0] for intensity, value[1] for phase)
    """
    labels_dense = loadmat(mat_file_name)["Lbl"]
    folder_names = labels_dense[:,0]
    intensity_labels = labels_dense[:,1]
    phase_labels = labels_dense[:,1] #labels_dense[:,2] 
    
    labels_dense_dict = {}
    
    for folder_idx in range(len(folder_names)):
        curr_folder_name = str(folder_names[folder_idx][0])
        tmp = curr_folder_name.split('-')[0]
        if tmp == '16' or tmp == '26' or tmp == '38':
            curr_folder_name = '0' + curr_folder_name  
            if tmp == '38':
                curr_folder_name = curr_folder_name.replace('SA','AS')
        elif tmp == '23':
            if curr_folder_name.split('-')[1] == 'PSM' and curr_folder_name.split('-')[-1] == '1.4':
                curr_folder_name = curr_folder_name.replace('1.4', '2.1')
        curr_inten_labels = intensity_labels[folder_idx].ravel()
        curr_phase_labels = phase_labels[folder_idx].ravel()
        labels_dense_dict.update({curr_folder_name: [curr_inten_labels, curr_phase_labels]}) 
    return labels_dense_dict



# def get_dense_FileLabel_Lists(root_dir, folders_dir, labels_path=args.label_dir, img_type='intensity'):
#     if img_type == 'intensity' or img_type == 'phase':
#         labels_dense_dict = read_labels_from_mat(labels_path)
#     else:
#         raise ValueError('Dense labels only support intensity/phase!')
    
#     folders_select = [os.path.join(root_dir, folder, img_type) for folder in folders_dir]
#     files_path = []
#     imgs_labels = []

#     for folder in folders_select:
#         folder_name_tmp = folder.split('/')[1]
#         folder_name = folder_name_tmp.split('_')[1]

#         if img_type == 'intensity':
#             curr_folder_labels = labels_dense_dict[folder_name][0]
#         elif img_type == 'phase':
#             curr_folder_labels = labels_dense_dict[folder_name][1] 
#         else:
#             raise ValueError('Dense labels only support intensity/phase!')
        
#         if folder_name == '026-S-1.1' or folder_name=='016-SA-1.1': ## two folders missing intensity/phase files
#             pass
#         else:
#             files_list = sorted(glob.glob(os.path.join(folder, '*.png')) + glob.glob(os.path.join(folder, '*.jpg')))
#             if not (len(files_list) == len(curr_folder_labels) ):
#                 raise ValueError("File list length not equal to label length, please check it!")
#             for file_idx, file_list in enumerate(files_list):
#                 files_path.append(file_list)
#                 imgs_labels.append(curr_folder_labels[file_idx])

#     return files_path, imgs_labels


def get_dense_FileLabel_Lists(root_dir, folders_dir, labels_path=args.label_dir, img_type='intensity'):
    if img_type == 'intensity' or img_type == 'phase':
        labels_dense_dict = read_labels_from_mat(labels_path)
    else:
        raise ValueError('Dense labels only support intensity/phase!')
    
    folders_select = [os.path.join(root_dir, folder, img_type) for folder in folders_dir]
    files_path = []
    imgs_labels = []

    for folder in folders_select:
        folder_name_tmp = folder.split('/')[1]
        folder_name = folder_name_tmp.split('_')[1]

        if img_type == 'intensity' or img_type == 'phase':
            curr_folder_labels = labels_dense_dict[folder_name][0]
        else:
            raise ValueError('Dense labels only support intensity/phase!')
        
        if folder_name == '026-S-1.1' or folder_name=='016-SA-1.1': ## two folders missing intensity/phase files
            pass
        else:
            files_list = sorted(glob.glob(os.path.join(folder, '*.png')) + glob.glob(os.path.join(folder, '*.jpg')))
            if not (len(files_list) == len(curr_folder_labels) ):
                raise ValueError("File list length not equal to label length, please check it!")
            for file_idx, file_list in enumerate(files_list):
                files_path.append(file_list)
                imgs_labels.append(curr_folder_labels[file_idx])

    return files_path, imgs_labels



folders = os.listdir(args.data_dir)
num_train_folder, num_val_folder = int(np.ceil(args.train_ratio * len(folders))), \
                        int(np.ceil(args.val_ratio * len(folders)))
num_test_folder = len(folders) - num_train_folder - num_val_folder
#print(num_train_folder, num_val_folder, num_test_folder) #96 (94 indeed, since 2 missing), 12, 11

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
imgs_train, labels_train = get_dense_FileLabel_Lists(args.data_dir, train_folders, img_type=args.img_type)
imgs_val, labels_val = get_dense_FileLabel_Lists(args.data_dir, val_folders, img_type=args.img_type)
imgs_test, labels_test = get_dense_FileLabel_Lists(args.data_dir, test_folders, img_type=args.img_type)
print('\ndata split: train/val/test=%d/%d/%d.\n' %(len(imgs_train),len(imgs_val),len(imgs_test)))  # train/val/test=23299/3008/2760

## shuffle training data
random.seed(args.rndSeed2)
seq = list(range(len(imgs_train)))
random.shuffle(seq)

imgs_train = [imgs_train[idx] for idx in seq]
labels_train = [labels_train[idx] for idx in seq]

#print(len(folders_Label))
#plt.hist(folders_Label, density=False, bins=50)  # density=False would make counts
#plt.ylabel('Count')
#plt.xlabel('Label');
#plt.show()


## create dataset
score_dataset_train = Human_Hip_Joint_Score(imgs_path=imgs_train, feats_dir=args.feats_dir, \
                                    imgs_label=labels_train, transform=transform_data, img_type=args.img_type) 
score_dataset_val = Human_Hip_Joint_Score(imgs_path=imgs_val, feats_dir=args.feats_dir, \
                                    imgs_label=labels_val, transform=transform_data, img_type=args.img_type)
score_dataset_test = Human_Hip_Joint_Score(imgs_path=imgs_test, feats_dir=args.feats_dir, \
                                    imgs_label=labels_test, transform=transform_data, img_type=args.img_type)
train_loader = DataLoader(score_dataset_train, batch_size=args.batch_size, shuffle=True,num_workers=args.num_worker)
val_loader = DataLoader(score_dataset_val, batch_size=args.batch_size, shuffle=True,num_workers=args.num_worker)
test_loader = DataLoader(score_dataset_test, batch_size=args.batch_size, shuffle=False,num_workers=args.num_worker)
dataloaders = {'train': train_loader, 'val':val_loader, 'test':test_loader}
dataset_sizes = {'train': len(train_loader), 'val': len(val_loader), 'test':len(test_loader)}
#print(dataset_sizes['test'])  #'train':83; 'val':24 ; 'test':22; num of epochs
#a = next(iter(dataloaders['train']))
#print(a['image'].shape)

def train_model(model, criterion, optimizer, scheduler, num_epochs=150):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    lowest_mae = 100000.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 50)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  
            else:
                model.eval()   

            running_loss = 0
            running_mae = 0

            for batch_data in dataloaders[phase]:
                inputs, labels = batch_data['image'], batch_data['label']
                inputs = inputs.to(device)
                labels = labels.float().to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs.squeeze(1), labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() 
                running_mae += torch.mean(torch.abs(outputs.squeeze(1) - labels))   

            scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_mae = running_mae.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f}, MAE: {:.4f}'.format(
                phase, epoch_loss, epoch_mae))

            if phase == 'val' and epoch_mae < lowest_mae:
                lowest_mae = epoch_mae
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since

    print('-' * 50)
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Lowest val MAE: {:4f}'.format(lowest_mae))

    model.load_state_dict(best_model_wts)
    return model


def test_model(pretrained_model, phase='test'):
    running_mae = 0
    for batch_data in dataloaders[phase]:
        inputs, labels = batch_data['image'], batch_data['label']
        inputs = inputs.to(device)
        labels = labels.float().to(device)
        with torch.no_grad():
            outputs = pretrained_model(inputs)
        running_mae += torch.mean(torch.abs(outputs.squeeze(1) - labels))
    mae = running_mae.double() / dataset_sizes[phase]

    print('-' * 50)
    print('Testing MAE: {:4f}'.format(mae))
    print('-' * 50)
       

if not os.path.isfile(ckpt_path_save):
    print("\nBegin training CNN: ")
    model.train()
    model = train_model(model, criterion, optimizer, exp_lr_scheduler,
                       num_epochs=args.epoch) 
    torch.save(model.state_dict(),ckpt_path_save)
    print("ckpt saved at: %s !\n" %(ckpt_path_save))

else:
    print("\nCkpt already exists, loading ...")
    model.load_state_dict(torch.load(ckpt_path_save))

## test 
model.eval()
test_model(model)
torch.cuda.empty_cache() 
sys.stdout.close()

#if args.img_type == 'feat_multi' or args.img_type == 'feat':
    #feats.close()


