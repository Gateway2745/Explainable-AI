import os
from datetime import datetime
import numpy as np
import tensorflow as tf
import gc

import train_loop

def shuffle(images, labels):
    """Return shuffled copies of the arrays, keeping the indexes of
    both arrays in corresponding places
    """

    cp_images = np.copy(images)
    cp_labels = np.copy(labels)

    rng_state = np.random.get_state()
    np.random.shuffle(cp_images)
    np.random.set_state(rng_state)
    np.random.shuffle(cp_labels)

    return cp_images, cp_labels
    
def split_train_and_test(images, labels, ratio=0.8):
    """Splits the array into two randomly chosen arrays of training and testing data.
    ratio indicates which percentage will be part of the training set."""

    images, labels = shuffle(images, labels)

    split = int(images.shape[0] * ratio)

    training_images = images[:split]
    training_labels = labels[:split]

    test_images = images[split:]
    test_labels = labels[split:]

    return [training_images, training_labels], [test_images, test_labels]

def train_single(inFile, size=512):
    """Train network a single time using the given files as input.
    inFile => path without extension (more than one file will be read)
    """

    print('Training...')

    # Load data
    images = np.load(inFile + '.npy', mmap_mode='r')
    labels = np.load(inFile + '_labels.npy', mmap_mode='r')

    # Create training and test sets
    training, test = split_train_and_test(images, labels)

    train_loop.train_net(training, test, size=size)