import os
import glob
import torch
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from scipy.io import loadmat
import h5py

def getFileList(root_dir, img_type='intensity'):
    folders = os.listdir(root_dir)
    folders_select = [os.path.join(root_dir, folder, img_type) for folder in folders]
    files_path = []
    for folder in folders_select:
        files_list = glob.glob(os.path.join(folder, '*.png')) + glob.glob(os.path.join(folder, '*.jpg'))
        for file_list in files_list:
            files_path.append(file_list)
        
    return files_path

    
def mergeChannels(img_pil_1, img_pil_2):
    img_pil_1_r, img_pil_1_g, img_pil_1_b = img_pil_1.split()
    img_pil_2_r, img_pil_2_g, img_pil_2_b = img_pil_2.split()
    img_pil_merge = Image.merge("RGB", (img_pil_1_r, img_pil_2_r, img_pil_1_b))
    return img_pil_merge
    
def processImage(inFilePath):
    image = Image.open(inFilePath)
    image = image.crop( (10, 0, 490,  360  ) )  #(480, 360)
    image = image.resize((240,180))
    return image   

class Human_Hip_Joint_Score(Dataset):
    def __init__(self, imgs_path, feats_dir, imgs_label, transform=None, img_type='intensity'):
        self.imgs_path = imgs_path
        self.feats_dir = feats_dir
        self.imgs_label = imgs_label 
        self.transform = transform
        self.img_type = img_type
        
    def __len__(self):
        return len(self.imgs_path)
        
    def __getitem__(self, index):
        img_path = self.imgs_path[index]
        img = Image.open(img_path)
        if self.img_type == 'multi' or 'feat_multi':
            img_path_phase = img_path.replace('intensity', 'phase')
            img_path_phase = img_path_phase.replace('OCT', 'PhaseRetard_bw')    
            img_phase = Image.open(img_path_phase)
            img = mergeChannels(img, img_phase)    
        img = img.crop( (10, 0, 490,  360  ) )  #(480, 360)   UBC_016-A-1.1_20181213_052_1202_0501_1020_PhaseRetard_bw1000.jpg
        label = self.imgs_label[index]
        
        sample = {'image': img, 'label': label}
        
        if self.transform:
            sample_tmp = self.transform(sample['image'])
        else:
            raise ValueError('Transform is needed.')
        if self.img_type == 'multi':                
            sample['image'] = sample_tmp[:2,:,:]
        elif self.img_type == 'feat_multi' or self.img_type == 'feat':
            img_name = img_path.split('/')[-1]
            #feat_path = self.feats_dir + img_name.split('_')[1] + '.mat'
            #feat = torch.Tensor( loadmat(feat_path)['data_comb'] )
            hf = h5py.File(self.feats_dir, 'r')
            feat_key = img_name.split('_')[1]
            feat = torch.Tensor( hf[feat_key] )
            feat = feat.permute(2,1,0)
            feat = feat.unsqueeze(0)
            feat = torch.nn.functional.interpolate(feat, size=(240, 180))
            feat = feat.squeeze(0)
            if self.img_type == 'feat_multi':
                sample['image'] = sample_tmp[:2,:,:]
                sample['image'] = torch.cat( (sample['image'], feat), 0) 
            else:
                sample['image'] = feat
        else:
            sample['image'] = sample_tmp[:1,:,:]
        
        return sample
        
        
       
class Human_Hip_Joint_Score_instance(Dataset):
    def __init__(self, imgs_path, feats_dir, imgs_label, transform=None, img_type='intensity'):
        self.imgs_path = imgs_path
        self.feats_dir = feats_dir
        self.imgs_label = imgs_label 
        self.transform = transform
        self.img_type = img_type
        
    def __len__(self):
        return len(self.imgs_path)
        
    def __getitem__(self, index):
        img_path = self.imgs_path[index]
        img = Image.open(img_path)
        if self.img_type == 'multi' or 'feat_multi':
            img_path_phase = img_path.replace('intensity', 'phase')
            img_path_phase = img_path_phase.replace('OCT', 'PhaseRetard_bw')    
            img_phase = Image.open(img_path_phase)
            img = mergeChannels(img, img_phase)    
        img = img.crop( (10, 0, 490,  360  ) )  #(480, 360)   UBC_016-A-1.1_20181213_052_1202_0501_1020_PhaseRetard_bw1000.jpg
        label = self.imgs_label[index]
        
        sample = {'image': img, 'label': label, 'name': img_path.split('/')[-1]}
        
        if self.transform:
            sample_tmp = self.transform(sample['image'])
        else:
            raise ValueError('Transform is needed.')
        if self.img_type == 'multi':                
            sample['image'] = sample_tmp[:2,:,:]
        elif self.img_type == 'feat_multi' or self.img_type == 'feat':
            img_name = img_path.split('/')[-1]
            #feat_path = self.feats_dir + img_name.split('_')[1] + '.mat'
            #feat = torch.Tensor( loadmat(feat_path)['data_comb'] )
            hf = h5py.File(self.feats_dir, 'r')
            feat_key = img_name.split('_')[1]
            feat = torch.Tensor( hf[feat_key] )
            feat = feat.permute(2,1,0)
            feat = feat.unsqueeze(0)
            feat = torch.nn.functional.interpolate(feat, size=(240, 180))
            feat = feat.squeeze(0)
            if self.img_type == 'feat_multi':
                sample['image'] = sample_tmp[:2,:,:]
                sample['image'] = torch.cat( (sample['image'], feat), 0) 
            else:
                sample['image'] = feat
        else:
            sample['image'] = sample_tmp[:1,:,:]
        
        return sample        
        