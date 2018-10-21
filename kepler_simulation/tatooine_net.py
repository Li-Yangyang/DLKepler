#CUDA_VISIBLE_DEVICES=0 python tatooine_net.py
import numpy as np
np.random.seed(1337)
from keras.models import Model
from keras.layers import Dense, Input, Activation, Dropout, Flatten, Bidirectional
from keras.layers import Convolution1D, MaxPooling1D, LSTM, BatchNormalization
from keras.preprocessing.sequence import pad_sequences
from keras import initializers
from keras.utils import np_utils
from keras.optimizers import RMSprop, Adam, SGD, Adadelta
from keras import backend as K
import tensorflow as tf
import sys
import pickle
import h5py
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score,roc_auc_score
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping,TensorBoard
from keras.models import model_from_json
import json
from keras import regularizers


def tatooine(data_shape):
    inputs = Input(shape=data_shape)#(X_train.shape[1], 1)

    layer = Convolution1D(filters=96, kernel_size=5, strides=1, kernel_initializer='glorot_uniform')(inputs)
    layer = BatchNormalization()(layer)
    layer = Activation('relu')(layer)

    layer = MaxPooling1D(pool_size=3)(layer)


    layer = Convolution1D(filters=256, kernel_size=5,strides=1,kernel_initializer='glorot_uniform')(layer)
    layer = BatchNormalization()(layer)
    layer = Activation('relu')(layer)


    layer = Convolution1D(filters=384, kernel_size=3,strides=1,kernel_initializer='glorot_uniform')(layer)
    layer = BatchNormalization()(layer)
    layer = Activation('relu')(layer)


    layer = Convolution1D(filters=384, kernel_size=3,strides=1,kernel_initializer='glorot_uniform')(layer)
    layer = BatchNormalization()(layer)
    layer = Activation('relu')(layer)


    layer = Convolution1D(filters=256, kernel_size=3,strides=1,kernel_initializer='glorot_uniform')(layer)
    layer = BatchNormalization()(layer)
    layer = Activation('relu')(layer)

    layer = MaxPooling1D(pool_size=3)(layer)


    layer = Flatten()(layer)
    layer = Dense(1024,kernel_initializer='glorot_uniform')(layer) #2048
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(1024,kernel_initializer='glorot_uniform')(layer) #2048
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(1)(layer)
    layer = Activation('sigmoid')(layer)


    return Model(inputs, layer)



def tatooine_new(data_shape):
    inputs = Input(shape=data_shape)#(X_train.shape[1], 1)

    layer = Convolution1D(filters=64, kernel_size=5, strides=1, kernel_initializer='glorot_uniform')(inputs)
    layer = BatchNormalization()(layer)
    layer = Activation('relu')(layer)

    layer = Convolution1D(filters=64, kernel_size=5, strides=1, kernel_initializer='glorot_uniform')(layer)
    layer = BatchNormalization()(layer)
    layer = Activation('relu')(layer)

    layer = MaxPooling1D(pool_size=7)(layer)


    layer = Convolution1D(filters=128, kernel_size=5, strides=1, kernel_initializer='glorot_uniform')(layer)
    layer = BatchNormalization()(layer)
    layer = Activation('relu')(layer)

    layer = Convolution1D(filters=128, kernel_size=5, strides=1, kernel_initializer='glorot_uniform')(layer)
    layer = BatchNormalization()(layer)
    layer = Activation('relu')(layer)

    layer = MaxPooling1D(pool_size=7)(layer)


    #layer = MaxPooling1D(pool_size=2)(layer)

    layer = Flatten()(layer)
    layer = Dense(1024,kernel_initializer='glorot_uniform')(layer) #2048
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(1024,kernel_initializer='glorot_uniform')(layer) #2048
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(1)(layer)
    layer = Activation('sigmoid')(layer)


    return Model(inputs, layer)

def tatooine_new2(data_shape):
    inputs = Input(shape=data_shape)#(X_train.shape[1], 1)

    layer = Convolution1D(filters=16, kernel_size=5, strides=1, kernel_initializer='glorot_uniform')(inputs)
    layer = BatchNormalization()(layer)
    layer = Activation('relu')(layer)

    layer = Convolution1D(filters=16, kernel_size=5, strides=1, kernel_initializer='glorot_uniform')(layer)
    layer = BatchNormalization()(layer)
    layer = Activation('relu')(layer)

    layer = MaxPooling1D(pool_size=7)(layer)

    layer = Convolution1D(filters=32, kernel_size=5, strides=1, kernel_initializer='glorot_uniform')(layer)
    layer = BatchNormalization()(layer)
    layer = Activation('relu')(layer)

    layer = Convolution1D(filters=32, kernel_size=5, strides=1, kernel_initializer='glorot_uniform')(layer)
    layer = BatchNormalization()(layer)
    layer = Activation('relu')(layer)

    layer = MaxPooling1D(pool_size=7)(layer)


    layer = Flatten()(layer)
    layer = Dense(512,kernel_initializer='glorot_uniform')(layer) #2048
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(512,kernel_initializer='glorot_uniform')(layer) #2048
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(1)(layer)
    layer = Activation('sigmoid')(layer)


    return Model(inputs, layer)

batch_size=32
epochs=50


trainspec_name= 'training_data_exo.hdf5'
with h5py.File(trainspec_name,'r') as hf:

    X_train=hf["training_lc"][:]
    y_train=hf["training_label"][:]
    info_train=hf["training_info"][:]

testspec_name='test_data_exo.hdf5'
with h5py.File(testspec_name,'r') as hf:
    X_test=hf["test_lc"][:]
    y_test=hf["test_label"][:]
    info_test=hf["test_info"][:]




y_train = y_train.reshape((-1, 1))
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

y_test = y_test.reshape((-1, 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))


for i in np.arange(10):
    print('training %s' % i)


    model = tatooine_new2((X_train.shape[1], 1))
    model.summary()
    optimization = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    #SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    logname='/scratch/Yinan/kepler_simulation/logs/tatooine_new_%slogs' % i
    tbCallBack = TensorBoard(log_dir=logname, histogram_freq=0, write_graph=True, write_images=True)

    """start training"""
    model.compile(loss='binary_crossentropy', optimizer=optimization, metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=7, verbose=1, mode='auto')
    tmp_name = '/scratch/Yinan/kepler_simulation/parameters/tatooine_new_'+str(i)+'.hdf5'
    model_checkpoint = ModelCheckpoint(tmp_name, monitor='val_loss', save_best_only=True)
    callbacks = [model_checkpoint,tbCallBack]
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.3, shuffle=True,
                        callbacks=callbacks)


    print("Validating...")
    scores = model.evaluate(X_test, y_test, verbose=0)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])
    print("Predicting...")

    y_pred = model.predict(X_test)
    y_pred=np.around(y_pred)

    print(accuracy_score(y_test, y_pred.round()) )
    print(precision_score(y_test, y_pred.round()) )
    print(recall_score(y_test, y_pred.round()) )
    print(f1_score(y_test, y_pred.round()) )
    print(roc_auc_score(y_test, y_pred.round()) )