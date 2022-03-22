#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 29 12:22:04 2021
Read dense labels (in .mat), and convert to dict format.
@author: yongwei
"""

from scipy.io import loadmat
import numpy as np

labels_dense=loadmat("Labels_dense.mat")['Lbl'] #(119,3)

folder_names = labels_dense[:,0]
intensity_labels = labels_dense[:,1]
phase_labels = labels_dense[:,2]

## get data structure
#folder_idx = 1
#img_idx = 10
#print(folder_names[fold_idx][0], intensity_labels[folder_idx][0,img_idx], phase_labels[folder_idx][0,img_idx])


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
    print("-"*20)
    print(curr_folder_name, np.mean(curr_inten_labels), np.mean(curr_phase_labels))  # check intensity/phase label values
    labels_dense_dict.update({curr_folder_name: [curr_inten_labels, curr_phase_labels]}) # add new key/values to dict
    ## double check
    print(curr_folder_name, np.mean(labels_dense_dict[curr_folder_name][0]), np.mean(labels_dense_dict[curr_folder_name][1]))
    print("-"*20)
#    
#
##rnd_test_name = "26-SA-1.2"
##print(np.mean(labels_dense_dict[rnd_test_name][0]), np.mean(labels_dense_dict[rnd_test_name][1]))
    
################## convert to function ########################
##
#def read_labels_from_mat(mat_file_name):
#    """Read dense labels from .mat file.
#    Input: mat_file_name, i.e., path to mat file.
#    Return: a dict with folder_name as keys, 
#            and intensity/phase arrays as values 
#            I.e., (value[0] for intensity, value[1] for phase)
#    """
#    labels_dense = loadmat(mat_file_name)["Lbl"]
#    folder_names = labels_dense[:,0]
#    intensity_labels = labels_dense[:,1]
#    phase_labels = labels_dense[:,2] 
#    
#    labels_dense_dict = {}
#    
#    for folder_idx in range(len(folder_names)):
#        curr_folder_name = str(folder_names[folder_idx][0])
#        tmp = curr_folder_name.split('-')[0]
#        if tmp == '16' or tmp == '26' or tmp == '38':
#            curr_folder_name = '0' + curr_folder_name  
#            if tmp == '38':
#                curr_folder_name = curr_folder_name.replace('SA','AS')
#        elif tmp == '23':
#            if curr_folder_name.split('-')[1] == 'PSM' and curr_folder_name.split('-')[-1] == '1.4':
#                curr_folder_name = curr_folder_name.replace('1.4', '2.1')
#        curr_inten_labels = intensity_labels[folder_idx].ravel()
#        curr_phase_labels = phase_labels[folder_idx].ravel()
#        labels_dense_dict.update({curr_folder_name: [curr_inten_labels, curr_phase_labels]}) 
#    return labels_dense_dict
#
#
#my_labels = read_labels_from_mat("Labels_dense.mat")

