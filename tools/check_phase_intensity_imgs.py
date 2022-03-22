import os
import glob
import csv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
from utils import getFileList


root_dir = 'data/'      
    

folders = os.listdir(root_dir)

def count_folder(folder_base, folder_name='phase'):
    folder_path = os.path.join(folder_base, folder_name)
    files_list = glob.glob(os.path.join(folder_path, '*.png')) + glob.glob(os.path.join(folder_path, '*.jpg'))
    return len(files_list)

cnt = 0
for folder in folders:
    folder_tmp = os.path.join(root_dir, folder) 
    num_phase_imgs = count_folder(folder_base=folder_tmp, folder_name='phase')
    num_intensity_imgs = count_folder(folder_base=folder_tmp, folder_name='intensity')
    #print("folder name:%s, num phase imgs:%d, num intensity imgs:%d" %(folder, num_phase_imgs, num_intensity_imgs)) 
    if num_phase_imgs != num_intensity_imgs:
        cnt += 1 
    
print()
print("num of folders that phase not equal intensity:%d"%(cnt))  # 0 
    

## dataset split
train_ratio, val_ratio = 0.8, 0.1
num_train_folder, num_val_folder = int(np.ceil(train_ratio * len(folders))), \
                        int(np.ceil(val_ratio * len(folders)))
                        
num_test_folder = len(folders) - num_train_folder - num_val_folder
#print(num_train_folder, num_val_folder, num_test_folder) #96, 12, 11

random.seed(2021)
seq_folder = list(range(len(folders)))
random.shuffle(seq_folder)

train_indx_folder, val_indx_folder = seq_folder[:num_train_folder], seq_folder[num_train_folder:num_train_folder+num_val_folder]
test_indx_folder = seq_folder[num_train_folder + num_val_folder:]


train_folders = [folders[idx] for idx in train_indx_folder] 
val_folders = [folders[idx] for idx in val_indx_folder]
test_folders = [folders[idx] for idx in test_indx_folder]


### get img list for each file
def getFileList_folders(root_dir, folders_dir, img_type='intensity'):
    folders_select = [os.path.join(root_dir, folder, img_type) for folder in folders_dir]
    files_path = []
    for folder in folders_select:
        files_list = glob.glob(os.path.join(folder, '*.png')) + glob.glob(os.path.join(folder, '*.jpg'))
        for file_list in files_list:
            files_path.append(file_list)       
    return files_path


    
imgs_train = getFileList_folders(root_dir, train_folders, img_type='intensity')
imgs_val = getFileList_folders(root_dir, val_folders, img_type='intensity')
imgs_test = getFileList_folders(root_dir, test_folders, img_type='intensity')
#print('data split: train/val/test=%d/%d/%d.\n' %(len(imgs_train),len(imgs_val),len(imgs_test)))  #data split: train/val/test=23299/3008/2760.

#get labels
imgs_label = []
folders_name = []
folders_Label = []
label_dir = 'Labels_v1.csv'
with open(label_dir, mode='r') as infile:
    reader = csv.DictReader(infile)
    for row in reader:
        folders_name.append(row['folderID'])
        folders_Label.append(float(row['folderLabel']))
    
folders_dict = dict(zip(folders_name, folders_Label))
def get_labels(imgs_path):
    imgs_label = [] 
    for img_path in imgs_path:
        folder_name_tmp = img_path.split('/')[1]
        folder_name = folder_name_tmp.split('_')[1]
        imgs_label.append(folders_dict[folder_name])
    return imgs_label

labels_train = get_labels(imgs_train)
labels_val = get_labels(imgs_val)
labels_test = get_labels(imgs_test)
#print(len(labels_train), len(labels_val), len(labels_test))
#test_idx = 1000
#print(imgs_train[test_idx], labels_train[test_idx])  #23299 3008 2760

## shuffle training data
random.seed(1)
seq = list(range(len(imgs_train)))
random.shuffle(seq)

imgs_train = [imgs_train[idx] for idx in seq]
labels_train = [labels_train[idx] for idx in seq]













