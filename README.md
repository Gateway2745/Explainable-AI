# Explaining Deep Learning algorithms used for classification of CXR images

### Requirements
Tensorflow==1.5<br/>
lungs-finder==1.0.0

###  Training on cheXpert  dataset

train from scratch
```python
1. download cheXpert dataset to project directory
2. python train.py --csv ./CheXpert-v1.0-small/train.csv --tensorboard-dir ./logs --checkpoint-dir ./snapshots
``` 
resume from checkpoint after loading weights
```python
python train.py --csv ./CheXpert-v1.0-small/train.csv --tensorboard-dir ./logs --checkpoint-dir ./checkpoints --load-weights ./checkpoints/model_3.hd5
```

###  Classification architecture 

 https://www.nature.com/articles/s41598-019-42557-4.pdf<br/>
 https://github.com/frapa/tbcnn
