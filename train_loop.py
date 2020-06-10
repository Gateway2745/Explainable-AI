import keras
from keras.metrics import (
        AUC
)
class Generator(keras.utils.Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        
    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size :(idx + 1) * self.batch_size]
        
        return batch_x,batch_y
        


def train_net(training, size=512, epochs=400, batch_size=4, logging_interval=5, run_name=None):
    
    x_train,y_train = training
    train_gen = Generator(x_train, y_train, 2)

    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy', AUC()])
    
    model.fit_generator(
        train_gen,
        epochs=5,
        )

