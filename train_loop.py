from keras.metrics import AUC
import net
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from preprocessing.generator import Generator
        
csv_file  = '~/CheXpert-v1.0-small/train.csv'

def train_net(size=512, epochs=20, batch_size=4, logging_interval=5, run_name=None):

    train_gen = Generator(csv_file, batch_size=batch_size)

    model = net.generate_network()

    print(model.summary())
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy', AUC()])
    
    model.fit_generator(
        train_gen,
        epochs=epochs,
        )

train_net()
