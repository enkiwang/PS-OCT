import os
import numpy as np
from scipy.io import loadmat
import glob
import h5py



mat_dir = 'feats'
h5_path = os.path.join(mat_dir, 'feat.h5')

files_list = glob.glob(os.path.join(mat_dir, '*.mat'))


f = h5py.File(h5_path, "w")
for feat_path in files_list:
    data_ = loadmat(feat_path)['data_comb']
    file_name_tmp = feat_path.split('.mat')[0]
    file_name_ = file_name_tmp.split('/')[-1]
    f.create_dataset(file_name_, data=data_, dtype=np.double)
    #print(file_name_)

f.close()


hf = h5py.File(h5_path, 'r')
print(list(hf.keys()))


for feat_key in list(hf.keys()):
    data = hf[feat_key]
    print(feat_key, data.shape)

hf.close()
