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
    output = inp
    
     for i in range(length):
        output = Conv2D(
            filters=filters,
            kernel_size=3,
            strides=stride,
            padding='same',
            kernel_initializer='he_normal'
        )(output)

        output = BatchNormalization()(output)       
        output = Activation('relu')(output)
        
    shortcut = Conv2D(
            filters=filters,
            kernel_size=1,
            strides=stride**length,
            padding='same',
            kernel_initializer='he_normal'
        )(inp)
    
    shortcut = BatchNormalization()(shortcut)
    shortcut = Activation('relu')(shortcut)
    
    output = (output + shortcut) / 2.
    
    if (pool):      
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
        output = GlobalAveragePooling2D(data_format='channels_last' name='gap')(output)
        
        output = Dense(
            units=2, # 2 outputs
            kernel_initializer='he_normal',
            name='logits'
        )(output)

        probabilities = Activation('softmax', name='probabilities')(output)


