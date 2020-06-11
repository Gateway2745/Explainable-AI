import os
import pandas as pd
import json

ATTR_NAMES = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
                  'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia',
                  'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other',
                  'Fracture', 'Support Devices']

DROP_COLS = ["Sex", "Age", "Frontal/Lateral", "AP/PA", "Fracture", "Support Devices"]
    
def preprocess_training_data(base_dir, data_filter = None, concat_train_val = False):
    
    train_file = os.path.join(base_dir, 'processed_train.csv')
    valid_file = os.path.join(base_dir, 'processed_valid.csv')
    
    if not (os.path.exists(train_file) and os.path.exists(valid_file)):
        # load data and preprocess training data
        valid_df = pd.read_csv(os.path.join(base_dir, 'valid.csv'), keep_default_na=True)
        train_df = load_training_data(os.path.join(base_dir, 'train.csv'), data_filter, valid_df, concat_train_val)

        # save
        train_df.to_csv(train_file) 
        valid_df.to_csv(valid_file)

        
def load_training_data(csv_path, data_filter, valid_df, concat_train_val):
    train_df = pd.read_csv(csv_path, keep_default_na=True)

    if(concat_train_val):
        train_df = pd.concat([train_df, valid_df])

    # 1. fill NAs (blanks for unmentioned) as 0 (negatives)
    train_df[ATTR_NAMES] = train_df[ATTR_NAMES].fillna(0)

    # 2. fill -1 as 1 (U-Ones method described in paper)
    train_df[ATTR_NAMES] = train_df[ATTR_NAMES].replace(-1,1)

    if data_filter is not None:
        # 3. apply attr filters
        # only keep data matching the attribute e.g. df['Frontal/Lateral']=='Frontal'
        for k, v in data_filter.items():
            train_df = train_df[train_df[k]==v]

        with open(os.path.join(os.path.dirname(csv_path), 'processed_training_data_filters.json'), 'w') as f:
            json.dump(data_filter, f)
       
    # 4. Delete irrelevant columns 
    if len(DROP_COLS)>0:
        train_df.drop(DROP_COLS, axis=1, inplace=True)
        
    return train_df    