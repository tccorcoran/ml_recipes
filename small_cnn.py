from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import ModelCheckpoint
from keras import regularizers

def small_cnn():
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(320, 240,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(.25))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(.25))
    
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(.25))
    
    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(64,kernel_regularizer=regularizers.l2(0.01)))
    model.add(Activation('relu'))
    model.add(Dropout(.5))
    model.add(Dense(32,kernel_regularizer=regularizers.l2(0.01)))
    model.add(Activation('relu'))
    model.add(Dense(5))
    model.add(Activation('softmax'))
    return model
