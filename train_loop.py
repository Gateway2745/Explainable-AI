from tensorflow.keras.metrics import AUC
from tensorflow.keras.callbacks import TensorBoard
import net
import argparse
import os
from preprocessing.generator import Generator
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def create_callbacks(args):
    callbacks = []

    if args['tensorboard_dir']:
        os.makedirs(args['tensorboard_dir'], exist_ok=True)
        tensorboard_callback = TensorBoard(
            log_dir                = args['tensorboard_dir'],
            histogram_freq         = 0,
            batch_size             = args['batch_size'],
            write_graph            = True,
            write_grads            = False,
            write_images           = False,
            embeddings_freq        = 0,
            profile_batch          = 0,
            embeddings_layer_names = None,
            embeddings_metadata    = None
        )
        
        callbacks.append(tensorboard_callback)
    return callbacks

csv_file  = '~/CheXpert-v1.0-small/train.csv'

def train_net(size=512, epochs=20, batch_size=4):
    
    ap = argparse.ArgumentParser()

    ap.add_argument('--batch-size', help='batch size', default=2)
    ap.add_argument('--tensorboard-dir', help='Log directory for Tensorboard output', default=False)
    
    args = vars(ap.parse_args())

    print('args is ', args)
    
    train_gen = Generator(csv_file, batch_size=args['batch_size'])
    
    callbacks = create_callbacks(args)

    model = net.generate_network()

    print(model.summary())
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy', AUC()])
    
    model.fit_generator(
        train_gen,
        epochs=epochs,
        callbacks=callbacks
        )

if __name__ == "__main__":
    train_net()
