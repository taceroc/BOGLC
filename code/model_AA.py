import numpy as np
import os
import pandas as pd
from config import *
import glob
from astropy.io import fits
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import time

train = np.zeros((549421,51,153))
test = np.zeros((183143,51,153))

train_targ = np.zeros((549421))
test_targ = np.zeros((183143))

train_ID = np.zeros((549421))
test_ID = np.zeros((183143))

init_len_tr = 0
init_len_te = 0
for ii in range(10):
    train_full= np.load('../data/data_split/train%d'%ii+'.npy') 
    test_full= np.load('../data/data_split/test%d'%ii+'.npy') 
    
    train_targ_full= np.load('../data/data_split/train_targ_%d'%ii+'.npy') 
    test_targ_full= np.load('../data/data_split/test_targ_%d'%ii+'.npy')
    
    train_ID_full= np.load('../data/data_split/train_ID_%d'%ii+'.npy') 
    test_ID_full= np.load('../data/data_split/test_ID_%d'%ii+'.npy') 
    
    np.random.seed(7)
    random_index_train = np.random.choice(len(train_full), size = len(train_full), replace=False)
    np.random.seed(3)
    random_index_test = np.random.choice(len(test_full), size=len(test_full), replace=False)

    #train = train_full
    #train_targ = train_targ_full
    #train_ID = train_ID_full

    #test = test_full
    #test_targ = test_targ_full
    #test_ID = test_ID_full

    trains = train_full[random_index_train,:,:]
    train[init_len_tr:init_len_tr+len(random_index_train)] = trains
    
    trains_targ = train_targ_full[random_index_train]
    train_targ[init_len_tr:init_len_tr+len(random_index_train)] = trains_targ
    
    trains_id = train_ID_full[random_index_train]
    train_ID[init_len_tr:init_len_tr+len(random_index_train)] = trains_id
    init_len_tr = init_len_tr + len(random_index_train)
    
    
    tests = test_full[random_index_test,:,:]
    test[init_len_te:init_len_te+len(random_index_test)] = tests
    
    tests_targ = test_targ_full[random_index_test]
    test_targ[init_len_te:init_len_te+len(random_index_test)] = tests_targ
    
    tests_id = test_ID_full[random_index_test]
    test_ID[init_len_te:init_len_te+len(random_index_test)] = tests_id
    
    init_len_te = init_len_te + len(random_index_test)
    

    train_full = None
    test_full = None
    train_targ_full = None
    test_targ_full = None
    train_ID_full = None
    test_ID_full = None
    trains = None
    tests = None
    trains_id = None
    tests_id = None
    
    print("Done with {}".format(ii))


times = 0
start = 0
end = 0
start = time.time()
#print('Start with %d'%ii+'\n')

np.random.seed(1)
tf.random.set_seed(346)

# -- define the network
layer1 = keras.layers.Conv2D(16, kernel_size=(5, 5), padding="valid", activation="relu", input_shape=(51, 153, 1))
layer2 = keras.layers.MaxPooling2D((2, 2), strides=2)
layer3 = keras.layers.Conv2D(32, kernel_size=(5, 5), padding="valid", activation="relu")
layer4 = keras.layers.MaxPooling2D((2, 2), strides=2)

layer5 = keras.layers.Conv2D(64, kernel_size=(5, 5), padding="valid", activation="relu")
layer6 = keras.layers.MaxPooling2D((2, 2), strides=2)

layer7 = keras.layers.Flatten()
# layer71 = keras.layers.Dropout(0.4)
# layer8 = keras.layers.Dense(64, activation="relu")
layer9 = keras.layers.Dense(32, activation="relu")
layer10 = keras.layers.Dense(2, activation="softmax")
layers = [layer1, layer2, layer3, layer4,layer5,layer6, layer7, layer9, layer10]

# -- instantiate the convolutional neural network
model = keras.Sequential(layers)

opt = keras.optimizers.SGD(learning_rate=0.01)
model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# define the checkpoint
filepath = "model_checkpoint_n.h5"
#checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='loss', verbose=1, 
 #                                               save_best_only=True, mode='min')
#callbacks_list = [checkpoint]

# --prevent that data is ordered by stamps folder
index_train = np.arange(len(train))
np.random.seed(7)
np.random.shuffle(index_train)
train = train[index_train,:,:]
train_targ = train_targ[index_train]
train_ID = train_ID[index_train]

index_test = np.arange(len(test))
np.random.seed(7)
np.random.shuffle(index_test)
test = test[index_test,:,:]
test_targ = test_targ[index_test]
test_ID = test_ID[index_test]


print(train.shape, test.shape)
print('unique test: {}'.format(np.unique(test_targ, return_counts=True)))
print('unique train: {}'.format(np.unique(train_targ, return_counts=True)))


# -- feautres need to have an extra axis on the end (for mini-batching)
feat_tr2 = train.reshape(len(train), 51, 153, 1)
feat_te2 = test.reshape(len(test), 51, 153, 1)

# -- fit the model
#history = model.fit(feat_tr2, train_targ, validation_split=0.20, epochs=4000, batch_size=2000,
                  # callbacks=callbacks_list)

# load the model
new_model = keras.models.load_model(filepath)

checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='loss', verbose=1, 
                                                save_best_only=True, mode='min')
callbacks_list = [checkpoint]
new_model.fit(feat_tr2, train_targ, validation_split=0.20, epochs=580, batch_size=2000,
                   callbacks=callbacks_list)

# -- print the accuracy
loss_tr, acc_tr = model.evaluate(feat_tr2, train_targ)
loss_te, acc_te = model.evaluate(feat_te2, test_targ)

print("Training accuracy : {0:.4f}".format(acc_tr))
print("Testing accuracy  : {0:.4f}".format(acc_te))

# print('Done {}'.format(ii))

end = time.time()
times  = (end - start)
print(times)

model_json = new_model.to_json()
with open("../outputs/model_model_AAA_n.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
new_model.save_weights("../outputs/model_model_AAA_n.h5")
print("Saved model to disk")
