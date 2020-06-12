import keras
import numpy as np
import cv2
import lungs_finder as lf
import os
from PIL import Image
import math
import random
import csv
from collections import OrderedDict
from preprocessing.preprocess import preprocess_training_data, DROP_COLS, PROCESSED_DIR

def _read_annotations(csv_reader):
    """ Parse the annotations file given by csv_reader.
    """ 
    result = OrderedDict()
    for line, row in enumerate(csv_reader):
        if(line==0):                #skip header
            continue 
        assert len(DROP_COLS) + len(row) == 20, "invalid input!"
        patient_id = row[0]
        img_path = row[1]
        attrs = [int(float(x)) for x in row[2:]]
        result[img_path] = {'id':patient_id, 'attrs':attrs}
        
    return result

def read_image_bgr(path):
    """ Read an image in BGR format.
    """
    image = np.ascontiguousarray(Image.open(path).convert('RGB'))
    
    return image[:, :, ::-1]

class Generator(keras.utils.Sequence):
    
    def __init__(
        self,
        csv_file,
        batch_size=2,
        shuffle_groups=True,
        image_min_side=512,
        image_max_side=800,
        base_dir=None,
        **kwargs
    ):
        
        self.batch_size             = int(batch_size)
        self.shuffle_groups         = shuffle_groups
        self.image_min_side         = image_min_side
        self.image_max_side         = image_max_side

        self.image_names = []
        self.image_data  = {}
        self.base_dir    = base_dir

        # Take base_dir from annotations file if not explicitly specified.
        if self.base_dir is None:
            self.base_dir = os.path.dirname(csv_file)

        #ignore lateral cxr images 
        preprocess_training_data(self.base_dir, data_filter = {'Frontal/Lateral':'Frontal'})
         
        train_csv = os.path.join(self.base_dir, PROCESSED_DIR, 'processed_train.csv')
        #valid_csv = os.path.join('base_dir', 'processed_valid.csv')

        try:
            with open(train_csv) as file:
                self.image_data = _read_annotations(csv.reader(file, delimiter=','))
        except ValueError as e:
            raise_from(ValueError('invalid CSV annotations file: {}: {}'.format(csv_file, e)), None)
            
        self.image_names = list(self.image_data.keys())
        
        # Define groups
        self.group_images()

        # Shuffle when initializing
        if self.shuffle_groups:
            self.on_epoch_end()
       
    def size(self):
        """ Size of the dataset.
        """
        return len(self.image_names)
          
    def group_images(self):
        """ Order the images according to self.order and makes groups of self.batch_size.
        """

        order = list(range(self.size()))
            
        random.shuffle(order)

        self.groups = [[order[x % len(order)] for x in range(i, i
                       + self.batch_size)] for i in range(0, len(order),
                       self.batch_size)]
    
    def extract_lungs(self, image_group):
        """  use lung-finder to extract lung region
        """
        new_group = []
        for img in image_group:
            found_lungs = lf.get_lungs(img)
            if found_lungs is not None and 0 not in found_lungs.shape:
                new_group.append(found_lungs)
            else:
                new_group.append(img)
        
        return new_group
    
    def on_epoch_end(self):
        random.shuffle(self.groups)
                  
    def __len__(self):
        return len(self.groups)
    
    def resize_image(self, img, min_side=512, max_side=800):
        """ resize images 
        """
        
        (rows, cols, _) = img.shape
        smallest_side = min(rows, cols)
        scale = min_side / smallest_side
        largest_side = max(rows, cols)
        if largest_side * scale > max_side:
            scale = max_side / largest_side

        img = cv2.resize(img, None, fx=scale, fy=scale)
        
        return img
    
    def preprocess_group_entry(self, image):
        """ Preprocess image and its annotations.
        """

        image = image.astype(np.float32)
            
        #TODO CALCULATE MEAN AND STD OF CHEXPERT DATA
        image /= 127.5
        image -= 1.

        # resize image
        image = self.resize_image(image, self.image_min_side, self.image_max_side)

        image = keras.backend.cast_to_floatx(image)

        return image
    
    def preprocess_group(self, image_group):
        """ Preprocess each image in its group.
        """
        
        for index in range(len(image_group)):
            image_group[index] = self.preprocess_group_entry(image_group[index])
    
        return image_group
    
    def load_annotations(self, image_index):
        """ Load annotations for an image_index.
        """
        path        = self.image_names[image_index]
        annotations = self.image_data[path]['attrs']
        
        return annotations

    def load_annotations_group(self, group):
        """ Load annotations for all images in group.
        """
        annotations_group = [self.load_annotations(image_index) for image_index in group]
    
        return annotations_group
    
    def image_path(self, image_index):
        """ Returns the image path for image_index.
        """
        return os.path.join(os.path.dirname(self.base_dir), self.image_names[image_index])
    
    def load_image(self, image_index):
        """ Load an image at the image_index.
        """
        return read_image_bgr(self.image_path(image_index))
    
    def load_image_group(self, group):
        """ Load images for all images in a group.
        """
        return [self.load_image(image_index) for image_index in group]
    
    def compute_inputs(self, image_group):
        """ Compute inputs for the network using an image_group.
        """
        max_shape = tuple(max(image.shape[x] for image in image_group) for x in range(3))

        image_batch = np.zeros((self.batch_size,) + max_shape, dtype=keras.backend.floatx())

        for image_index, image in enumerate(image_group):
            image_batch[image_index, :image.shape[0], :image.shape[1], :image.shape[2]] = image

        return image_batch

    def compute_input_output(self, group):
        """ Compute inputs and target outputs for the network.
        """
        # load images and annotations
        image_group = self.load_image_group(group)

        # extract lung regions  
        image_group = self.extract_lungs(image_group)

        # perform preprocessing steps
        image_group = self.preprocess_group(image_group)

        # compute network inputs
        inputs = self.compute_inputs(image_group)
        
        #compute targets 
        targets = np.array(self.load_annotations_group(group))
        
        return inputs,targets

    def __getitem__(self, index):
        
        group = self.groups[index]        
        inputs,targets = self.compute_input_output(group)
        
        return inputs, targets
 