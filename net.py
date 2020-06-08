from keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Dense,
    GlobalAveragePooling2D,
    Input,
    MaxPooling2D,
)

def generate_convolutional_block(inp, filters, length=2, pool=True, stride=1):
    "Generates a convolutional block, with a couple of simple options"

    output = inp
    
    for i in range(length):
        # convolution
        output = Conv2D(
            filters=filters,
            kernel_size=3,
            strides=stride,
            padding='same',
            kernel_initializer='he_normal'
        )(output)

        # batch normalization
        output = BatchNormalization()(output)

        # ReLU       
        output = Activation('relu')(output)
        
    shortcut = Conv2D(
            filters=filters,
            kernel_size=1,
            strides=stride**length,
            padding='same',
            kernel_initializer='he_normal'
        )(inp)
    
    # batch normalization
    shortcut = BatchNormalization()(shortcut)

    # ReLU
    shortcut = Activation('relu')(shortcut)
    
    output = (output + shortcut) / 2.
    
    if pool:      
        output = MaxPooling2D(
            pool_size=3,
            strides = 2
        )(output)
    
    return output

    def generate_network(size=512, width=1):
        inp = Input(shape = (size,size,1), name='input')
        
        output = generate_convolutional_block(inp, filters=16*width, stride=2)

        # 3 "normal" convolutional blocks
        output = generate_convolutional_block(output, filters=32*width)
        output = generate_convolutional_block(output, filters=48*width)
        output = generate_convolutional_block(output, filters=64*width)

        # last convolutional block without pooling
        output = generate_convolutional_block(output, filters=80*width, pool=False)

        # Global average pooling
        output = GlobalAveragePooling2D(data_format='channels_last')(output)
        
        output = Dense(
            units=2, # 2 outputs
            kernel_initializer='he_normal',
            name='logits'
        )(output)

        probabilities = Activation('softmax', name='probabilities')(output)

        classes = keras.argmax(logits, axis=1, name='classes')

        return inp, labels, {
        'logits': logits,
        'probabilities': probabilities,
        'classes': classes,
    }

def generate_functions(inp, labels, output):
    """Generates functions like error, accuracy and train, 
    that are used for training and testing the network"""

    #error = tf.losses.sparse_softmax_cross_entropy(labels, output['logits'])
    error = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=output['logits'])

    #optimizer = tf.train.AdamOptimizer(learning_rate=8e-5)
    optimizer = keras.optimizers.Adam(decay=8e-5)
    train = optimizer.minimize(error)

    metrics = {
        'accuracy': tf.metrics.accuracy(labels, output['classes']),
        'AUC': tf.metrics.auc(labels, output['probabilities'][:,1]),
    }

    return error, train, metrics
