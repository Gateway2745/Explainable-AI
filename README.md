# Explaining Deep Learning algorithms used for classification of CXR images

### Requirements
Tensorflow==1.15<br/>
Keras==2.3.1<br/>
Python>=3.6<br/>
lungs-finder==1.0.0

###  Training on cheXpert  dataset

train from scratch
```python
1. download cheXpert dataset to project directory
2. python train.py --csv ./CheXpert-v1.0-small/train.csv --tensorboard-dir ./logs --checkpoint-dir ./snapshots --gpu 1
``` 
resume from checkpoint after loading weights
```python
python train.py --csv ./CheXpert-v1.0-small/train.csv --tensorboard-dir ./logs --checkpoint-dir ./checkpoints --load-weights ./checkpoints/model_3.hd5 --gpu 1
```
###  Preprocessing techniques
1. RGB images converted to gray scale.
2. lungs-finder to extract lung regions.
3. histogram equalization.
4. input images resized such that minimum side is equal to 512px and maximum side is less than 800px.

###  Architecture Details
1.	 12 classes out of 14 are used to train our network. 
2.	Only frontal CXR images are retained ( Lateral not considered).
3. Multi-label classifcation with sigmoid activation for output layer and binary-crossentropy loss.

###  Classification architecture 

 https://www.nature.com/articles/s41598-019-42557-4.pdf<br/>
 https://github.com/frapa/tbcnn
