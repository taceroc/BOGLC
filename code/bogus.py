import numpy as np
import os
import pandas as pd
from config import *
import glob
from astropy.io import fits
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import seaborn as sns

def created_path():
    pttype = '*'
    flist = []
#     for i in range(1):
#         path = os.path.join(configs["dpath"],'stamps%d'%i,'SNWG','Archive','*','Y1','*','*',pttype + '*.fits')
#         flist.append(sorted(glob.glob(path)))
    #path10 = os.path.join(configs["dpath"],'stamps10','*',pttype + '*.fits')
    #flist.append(sorted(glob.glob(path10)))                      
    #print(len(flist))
    for i in ["20130829","20130831", "20130901"]:
        path = os.path.join(configs["dpath"],'stamps1','SNWG','Archive','*','Y1',i,'*',pttype + '*.fits')
        flist.append(sorted(glob.glob(path)))
    return np.concatenate((flist))

def return_ids(flist):
    return [int(f.split('/')[-1][4:-5]) for f in flist]

def DF_current_ids(ID):
    ffpath = os.path.join(configs["dpath"], "autoscan_features.3.feather") #this .feather file contain only the ID and OBJECT_TYPE for the images that I have on 
    new_labels = pd.read_feather(ffpath)
    current_labels = new_labels[new_labels["ID"].isin(ID)]
    current_labels = current_labels[["ID", "OBJECT_TYPE"]]
    current_labels.drop_duplicates(inplace=True) 
    current_labels = current_labels.sort_values(by= ["ID"]).reset_index(drop=True)
    counts_type = np.unique(current_labels['OBJECT_TYPE'], return_counts=True)
    #how_many = {"Real (0)":counts_type[1][0], "Bogus (1)": counts_type[1][1] }
    
    if len(counts_type[0]) == 2:
        print("Real (0) = {} and Bogus (1) = {}".format(counts_type[1][0], counts_type[1][1]))
    if len(counts_type[0]) == 1:
        if counts_type[0] == 0:
            print("Real (0) = {}".format(counts_type[0][0]))
        else:
            print("Bogus (1) = {}".format(counts_type[0][0]))
    return current_labels

def open_fits(current_labels, flist):
    imlist_dict = {}
    
    # stores the name of the images as a list for ID above
    #is a circle because i extract the ID for the flist, buttt
    imlist_dict['flist'] = [f for f in flist if int(f.split('/')[-1][4:-5]) in current_labels['ID'].to_numpy()]
    #print (len(imlist_dict['flist']))
    #print(flist.nbytes)
    del(flist)
    imlist_dict["imshp"] = fits.open((imlist_dict["flist"][0]))[0].data.shape #shape row,col
    extension="fits"
    imdtype = {"fits":float, "gif":np.uint8, }
    
    #sort as: descending ID and diff, srch, temp
    imlist_dict["flist"] = sorted(imlist_dict["flist"], key=lambda s: s.split('/')[-1][:4])
    imlist_dict["flist"]= sorted(imlist_dict["flist"], key=lambda s: s.split('/')[-1][4:])
    
    #container for data train and data test
    data_full = np.zeros((len(imlist_dict["flist"]),imlist_dict["imshp"][0], imlist_dict["imshp"][1]),imdtype[extension])

    #fill the container and open images
    for i in range(len(imlist_dict["flist"])):
    #for i in range(2):
        datas = fits.open(''.join(imlist_dict["flist"][i]), memmap=True)
        #datas.close()
        data_full[i] = datas[0].data
        #print("{}, path:{}".format(i,imlist_dict["flist"][i]))
        datas.close()
    del(imlist_dict)
    return data_full

def norm_data(data_full):
    data_norm = data_full.astype(float)
    data_full = None
    # --normalize
    # mean and std for diff images
    # min and max for srch and temp

    data_norm[::3] = (data_norm[::3]- data_norm[::3].mean(axis=(1,2), keepdims=True))/data_norm[::3].std(axis=(1,2), keepdims=True) #diff
    data_norm[1::3]= (data_norm[1::3]-data_norm[1::3].min(axis=(1,2), keepdims=True))/data_norm[1::3].max(axis=(1,2), keepdims=True) #srch
    data_norm[2::3]= (data_norm[2::3]-data_norm[2::3].min(axis=(1,2), keepdims=True))/data_norm[2::3].max(axis=(1,2), keepdims=True) #temp
    #print(data_full.nbytes)
    #del(data_full)
    #print(data_full.nbytes)
    return data_norm

def concatenate_normdata(data_norm):
    #concatenate diff srch temp for the same ID

    #final_data = np.zeros((int(len(data_norm)//3),data_norm.shape[1], data_norm.shape[1]*3))
    final_data = np.concatenate((data_norm[::3],data_norm[1::3],data_norm[2::3]), axis = 2)
    #print(data_norm.nbytes)
    #del(data_norm)
    #print(data_norm.nbytes)
    return final_data

def separate_types(current_labels):
    #exxtract the objects  = 0
    df_ID_0 = current_labels[current_labels["OBJECT_TYPE"]==0]
    #exxtract the objects  = 1
    df_ID_1 = current_labels[current_labels["OBJECT_TYPE"]==1]
    
    return df_ID_0,df_ID_1

def balance_data(df_ID_0, df_ID_1, final_data):
    len_each_set = min(len(df_ID_0), len(df_ID_1))
    if len(df_ID_0) <= len_each_set:
        #extract random the number of data classify as 0
        index_data_ID0 = df_ID_0.sample(len_each_set-10, random_state = 2).sort_index()
        #extract random the number of data classify as 1
        index_data_ID1 = df_ID_1.sample(len_each_set+10,random_state = 2).sort_index()
    else:
        #extract random the number of data classify as 0
        index_data_ID0 = df_ID_0.sample(len_each_set+10, random_state = 2).sort_index()
        #extract random the number of data classify as 1
        index_data_ID1 = df_ID_1.sample(len_each_set-10,random_state = 2).sort_index()
    
    #convert index to numpy to iterate
    index_ID0 = index_data_ID0.index.to_numpy()
    index_ID1 = index_data_ID1.index.to_numpy()
    
    #concatenate both index
    indexes = sorted(np.concatenate((index_ID0, index_ID1)))
    
    equal_type_data = np.array([final_data[i] for i in indexes])
    return indexes, equal_type_data

def split_data(equal_type_data, final_data, indexes): #equal_type_data is already balance
    #70% is for training
    #30% testing
    train_len = int(equal_type_data.shape[0]*0.7)
    test_len = equal_type_data.shape[0]  - int(equal_type_data.shape[0]*0.7)
    
    #random data
    import random
    random.seed(4)
    random_index = random.sample(range(0, equal_type_data.shape[0]), train_len)
    
    train = np.array([final_data[i] for i in [indexes[i] for i in sorted(random_index)]])
    test = np.array([final_data[i] for i in indexes if i not in [indexes[i] for i in sorted(random_index)]])
    #del(final_data)
    return train, test, random_index


def targets(current_labels, indexes, random_index):
    #extracting the label 0 or 1
    targets = [current_labels.iloc[i]["OBJECT_TYPE"] for i in indexes]
    #split the targets
    train_targ = np.array([current_labels.iloc[i]["OBJECT_TYPE"] for i in [indexes[i] for i in sorted(random_index)]])
    test_targ = np.array([current_labels.iloc[i]["OBJECT_TYPE"] for i in indexes if i not in [indexes[i] for i in sorted(random_index)]])
    
    del(indexes)
    del(random_index)
    return train_targ, test_targ

def keras_model_and_save(train, test, train_targ, test_targ ):
    np.random.seed(1)
    tf.random.set_seed(346)

    # -- define the network
    layer1 = keras.layers.Conv2D(16, kernel_size=(5, 5), padding="valid", activation="relu", input_shape=(51, 153, 1))
    layer2 = keras.layers.MaxPooling2D((2, 2), strides=2)
    layer3 = keras.layers.Conv2D(32, kernel_size=(5, 5), padding="valid", activation="relu", input_shape=(51, 153, 1))
    layer4 = keras.layers.MaxPooling2D((2, 2), strides=2)
    layer5 = keras.layers.Flatten()
    #layer6 = keras.layers.Dropout(0.4)
    layer7 = keras.layers.Dense(32, activation="relu")
    layer8 = keras.layers.Dense(2, activation="softmax")
    layers = [layer1, layer2, layer3, layer4, layer5, layer7,layer8]

    # -- instantiate the convolutional neural network
    model = keras.Sequential(layers)

    opt = keras.optimizers.SGD(learning_rate=0.01)
    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # -- feautres need to have an extra axis on the end (for mini-batching)
    feat_tr2 = train.reshape(len(train), 51, 153, 1)
    feat_te2 = test.reshape(len(test), 51, 153, 1)

    # -- fit the model
    history = model.fit(feat_tr2, train_targ, validation_split=0.20, epochs=40, batch_size=20, verbose = 0)

    # -- print the accuracy
    loss_tr, acc_tr = model.evaluate(feat_tr2, train_targ)
    loss_te, acc_te = model.evaluate(feat_te2, test_targ)

    print("Training accuracy : {0:.4f}".format(acc_tr))
    print("Testing accuracy  : {0:.4f}".format(acc_te))

#     model_json = model.to_json()
#     with open("model_small.json", "w") as json_file:
#         json_file.write(model_json)
#     # serialize weights to HDF5
#     model.save_weights("model_small.h5")
#     print("Saved model to disk")
    
    return loss_tr, acc_tr, loss_te, acc_te, history, model

def load_model():
    np.random.seed(1)
    tf.random.set_seed(346)

    # load json and create model
    json_file = open('model_test.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = keras.models.model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model_test.h5")
    return loaded_model
def extract_layer(flist, df_ID_1, model):
    
    bogus_full = open_fits(df_ID_1, flist)
    bogus_norm = norm_data(bogus_full)
    bogus_final = concatentate_normdata(bogus_norm)

    # -- feautres need to have an extra axis on the end (for mini-batching)
    bogus_reshape = bogus_final.reshape(int(len(bogus_full)//3), 51, 153, 1)
    
    # extract the hidden layer for layer6 = Dense(32)
    get_6rd_layer_output = tf.keras.backend.function([model.layers[0].input],
                                      [model.layers[5].output])
    #input = 0 type
    layer_output = get_6rd_layer_output([bogus_reshape])
   
    return bogus_full,layer_output


flist = created_path()
print(flist)
print(len(flist))
#print(flist[-10:])
ID = return_ids(flist)
current_labels = DF_current_ids(ID)
#print(current_labels)
#print(current_labels[0]['ID'])
data_full = open_fits(current_labels,flist)
print(len(data_full))
#print(data_full)
#plt.imshow(data_full[0])
#plt.savefig('lol.png')
#plt.show()
data_norm = norm_data(data_full)
#data_full = None
final_data = concatenate_normdata(data_norm)
data_norm = None
df_ID_0, df_ID_1 = separate_types(current_labels)
indexes, equal_type_data = balance_data(df_ID_0, df_ID_1, final_data)
train, test,random_index = split_data(equal_type_data,final_data, indexes)
final_data = None
train_targ, test_targ = targets(current_labels, indexes, random_index)
loss_tr, acc_tr, loss_te, acc_te, history, model = keras_model_and_save(train, test, train_targ, test_targ)

# -- plot the loss function
fig, ax = plt.subplots(figsize=(8, 5))
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.legend(["train", "test"], loc="upper left")
ax.set_xlabel("epoch", fontsize=15)
ax.set_ylabel("loss", fontsize=15)
#plt.savefig("loss_small_1.pdf",bbox_inches="tight")

# -- plot the accuracy function
fig, ax = plt.subplots(figsize=(8, 5))
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.legend(["train", "test"], loc="upper left")
ax.set_xlabel("epoch", fontsize=15)
ax.set_ylabel("acc", fontsize=15)
#plt.savefig("acc_small_1.pdf",bbox_inches="tight")

# #confusion matrix for test data
# feat_tr2 = train.reshape(len(train), 51, 153, 1)
# feat_te2 = test.reshape(len(test), 51, 153, 1)
# y_pred = model.predict(feat_te2)
# con_mat = tf.math.confusion_matrix(labels=test_targ, predictions=np.argmax(y_pred,axis=1)).numpy()
# con_mat.flatten()
# con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
 
# con_mat_df = pd.DataFrame(con_mat_norm,
#                      index = [0,1], 
#                      columns = [0,1])
# con_mat_norm.flatten()

# labels = [f"{v1}\n{v2*100}%" for v1, v2 in zip(con_mat.flatten(),con_mat_norm.flatten())]

# labels = np.asarray(labels).reshape(2,2)
# categories = ["0: Real", "1: Bogus"]

# figure = plt.figure(figsize=(4, 5))
# sns.heatmap(con_mat, annot=labels,cbar=False,fmt='',xticklabels=categories,yticklabels=categories,cmap='Pastel2_r')
# plt.tight_layout()
# plt.ylabel('True label')
# plt.xlabel('Predicted label')
# plt.savefig("confusionmatrix_small_GPUGPU.pdf",bbox_inches="tight")
# plt.show()