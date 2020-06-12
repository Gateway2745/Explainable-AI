# Explaining Deep Learning algorithms used for classification of CXR images

### Requirements
Tensorflow==1.5<br/>
lungs-finder==1.0.0

###  Training on cheXpert  dataset

```python
1. download cheXpert dataset to project directory
2. python --csv ./CheXpert-v1.0-small/train.csv --tensorboard-dir ./logs --checkpoint-dir ./snapshots
``` 

###  Classification architecture 

 https://www.nature.com/articles/s41598-019-42557-4.pdf<br/>
 https://github.com/frapa/tbcnn
