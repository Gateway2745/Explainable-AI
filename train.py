from keras.metrics import AUC
from keras.callbacks import Callback,TensorBoard
from keras.models import load_model
from setup_gpu import setup_gpu
import net
import argparse
import os
from preprocessing.generator import Generator
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class CheckpointSaver(Callback):
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir
    
    def on_epoch_end(self, epoch, logs={}):
        self.model.save_weights(os.path.join(self.checkpoint_dir, "model_{}.hd5").format(epoch+1))
                
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
            embeddings_layer_names = None,
            embeddings_metadata    = None
        )       
        callbacks.append(tensorboard_callback)
        
        
    if args['checkpoint_dir']:
        os.makedirs(args['checkpoint_dir'], exist_ok = True)
        checkpoint = CheckpointSaver(args['checkpoint_dir'])
        
        callbacks.append(checkpoint)
        
    return callbacks
        
def train_net():
    
    ap = argparse.ArgumentParser()

    ap.add_argument('--batch-size', help='batch size', default=2)
    ap.add_argument('--tensorboard-dir', help='Log directory for Tensorboard output', default=False)
    ap.add_argument('--checkpoint-dir', help='directory to save checkpoints',default=False)
    ap.add_argument('--epochs', help='number of epochs to train the model', default=10)
    ap.add_argument('--load-weights', help='load_weights from checkpoint', default=None)
    ap.add_argument('--gpu', help='gpu id', default=None)
    ap.add_argument('--csv', help='path to chexpert train.csv', required=True)
    
    args = vars(ap.parse_args())
    
    train_gen = Generator(args['csv'], batch_size=args['batch_size'])
    
    callbacks = create_callbacks(args)

    model = net.generate_network()
    
    print(model.summary())
    
    if args['gpu'] is not None:
        setup_gpu(int(args['gpu']))
        
    if args['load_weights'] is not None:
        print('Loading weights...')
        model.load_weights(args['load_weights'])
        initial_epoch = int(list(filter(str.isdigit, args['load_weights']))[0])
    else:  
        initial_epoch = 0
    
    model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['binary_accuracy', AUC(name='auc')])
    
    model.fit_generator(
        train_gen,
        epochs=int(args['epochs']),
        callbacks=callbacks,
        initial_epoch=initial_epoch
        )

if __name__ == "__main__":
    train_net()
