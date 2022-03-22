# PS-OCT
Polarization-sensitive optical coherence tomography 

This repository provides codes to "Polarization-sensitive optical coherence tomography with deep learning for detecting the local distribution of osteoarthritis severities". 

## Regression 

In regression case, we experiment on two types of regression labels: coarse labels and dense labels.

For coarse label regression:
```python
GPU_ID=0
python main.py --img_type='phase' --epoch=150 --gpu_id=${GPU_ID} --model_select='vgg16' --step_size 20 --gamma 0.2
python main.py --img_type='phase' --epoch=150 --gpu_id=${GPU_ID} --model_select='resnet18' --step_size 20 --gamma 0.2
python main.py --img_type='phase' --epoch=150 --gpu_id=${GPU_ID} --model_select='densenet121' --step_size 20 --gamma 0.2
python main.py --img_type='phase' --epoch=150 --gpu_id=${GPU_ID} --model_select='mobilenetv2' --step_size 20 --gamma 0.2
```

Similarly, you can experiment on "intensity" images by changing "phase" to "intensity".


For dense label regression:

```python
GPU_ID=0
python main_dense.py --img_type='phase' --epoch=150 --gpu_id=${GPU_ID} --model_select='vgg16' --step_size 20 --gamma 0.2
python main_dense.py --img_type='phase' --epoch=150 --gpu_id=${GPU_ID} --model_select='resnet18' --step_size 20 --gamma 0.2
python main_dense.py --img_type='phase' --epoch=150 --gpu_id=${GPU_ID} --model_select='densenet121' --step_size 20 --gamma 0.2
python main_dense.py --img_type='phase' --epoch=150 --gpu_id=${GPU_ID} --model_select='mobilenetv2' --step_size 20 --gamma 0.2
```


## Classification

For PS-OCT classification:

```python
GPU_ID=0
python main_class.py --img_type='phase' --num_class 2 --epoch=150 --gpu_id=${GPU_ID} --model_select='vgg16' --step_size 7 --gamma 0.2
python main_class.py --img_type='phase' --num_class 2 --epoch=150 --gpu_id=${GPU_ID} --model_select='resnet18' --step_size 7 --gamma 0.2
python main_class.py --img_type='phase' --num_class 2 --epoch=150 --gpu_id=${GPU_ID} --model_select='densenet121' --step_size 7 --gamma 0.2
python main_class.py --img_type='phase' --num_class 2 --epoch=150 --gpu_id=${GPU_ID} --model_select='mobilenetv2' --step_size 7 --gamma 0.2
```
