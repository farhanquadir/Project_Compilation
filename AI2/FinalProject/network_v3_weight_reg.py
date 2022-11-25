import keras
from keras.processing.text import Tokenizer
from keras.models import Sequential, load_model
from keras.utils import np_utils,to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, LSTM
from keras import regularizers
from keras.callbacks import LearningRateScheduler
from keras.optimizers import Adam, Adadelta, SGD, RMSprop, Adagrad, Adamax, Nadam
from keras.regularizers import l2
import tensorflow as tf
import numpy as np
import os,sys
import pandas as pd
 
def lr_schedule(epoch):
    lrate = 0.001
    if epoch > 75:
        lrate = 0.0005
    if epoch > 100:
        lrate = 0.0003
    return lrate

def readListFile(list_file):
    file_list=[]
    with open (list_file,"r") as f:
        for line in f:
            file_list.append(line.strip())
    return file_list

def loadData(datafile):
    data=pd.read_csv(datafile,header=None)
    mean_X=data.mean()[0]
    mean_Y=data.mean()[1]
    mean_Z=data.mean()[2]
    var_X=data.var()[0]
    var_Y=data.var()[1]
    var_Z=data.var()[2]
    stdev_X=data.std()[0]
    stdev_Y=data.std()[1]
    stdev_Z=data.std()[2]
    data[3]=mean_X
    data[4]=mean_Y
    data[5]=mean_Z
    data[6]=var_X
    data[7]=var_Y
    data[8]=var_Z
    data[9]=stdev_X
    data[10]=stdev_Y
    data[11]=stdev_Z
    #print(data)
    #sys.exit()
    #data=np.loadtxt(datafile,delimiter=",")
    data=data.to_numpy()
    data=normalize(data)
    #print (data.shape)
    #sys.exit()
    label=int(datafile.replace(".csv","").split("-")[-1])
    xx,yy=data.shape
    dat=np.zeros((660,12,1))
    #dat=np.zeros((660,3))
    dat[0:xx,0:yy,0]=data
    #print (len(dat))
    #sys.exit()
    #dat[0:xx,0:yy]=data
    """
    if (label==0):
        label=np.zeros(2)
        label[0]=1
    else:
        label=np.zeros(2)
        label[1]=1
    """
    return dat,label

def normalize(arr):
    mini=np.min(arr[0])
    maxi=np.max(arr[0])
    arr[0]=(arr[0]-mini)/(maxi-mini)
    mini=np.min(arr[1])
    maxi=np.max(arr[1])
    arr[1]=(arr[1]-mini)/(maxi-mini)
    mini=np.min(arr[2])
    maxi=np.max(arr[2])
    arr[2]=(arr[2]-mini)/(maxi-mini)
    return arr

def normalize_2(arr):
    mini=np.min(arr)
    maxi=np.max(arr)
    arr=arr/maxi
    return arr

def createModel(x_train):
    model=Sequential()
    model.add(LSTM(5000, return_sequences=True, input_shape=(),activation="relu"))
    
    return model

def saveModel(model):
    model.save(sys.argv[0].replace(".py","")+"HW4.h5")
    model.save_weights(sys.argv[0].replace(".py","")+"HW4_weights.h5")
    return

def trainModel(model,X_train,Y_train,X_test,Y_test,epoch=1,batch_size=1):
    optim = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    #model.compile(loss="categorical_crossentropy", optimizer=optim, metrics=["accuracy"])
    model.compile(loss="binary_crossentropy", optimizer=optim, metrics=["accuracy"])
    #history=model.fit(X_train, Y_train, validation_data=(X_test,Y_test), epochs= epoch, batch_size=batch_size)
    history=model.fit(X_train, Y_train, validation_split=0.2, epochs= epoch, batch_size=batch_size)
    saveModel(model)
    return history

def testModel(X_test,Y_test,model=None):
    if (model==None):
        model=load_model(sys.argv[0].replace(".py","")+"HW4.h5")
    
    
    return

#Read the list of training and test data
train_data_folder="./arrangedTrainingData/"#os.path.abstpath("")
test_data_folder="./arrangedTestData/"
training_list_file="training_list.txt"
test_list_file="test_list.txt"
training_file_list=readListFile(training_list_file)
test_file_list=readListFile(test_list_file)

n_train=len(training_file_list)
n_test=len(test_file_list)

#The following loads the training data and labels into a list. Labels are one-hot encoded
xyz_train_list=[]
label_train_list=[]
for i in range(n_train):
    data,label=loadData(train_data_folder+training_file_list[i])
    xyz_train_list.append(data)
    label_train_list.append(label)

#The following loads the test data and labels into a list. Labels are one-hot encoded
xyz_test_list=[]
label_test_list=[]
for i in range(n_test):
    data,label=loadData(test_data_folder+test_file_list[i])
    xyz_test_list.append(data)
    label_test_list.append(label)

xyz_train_list=np.asarray(xyz_train_list)

label_train_list=np.asarray(label_train_list)
print (label_train_list.shape)
label_train_list=to_categorical(np.asarray(label_train_list))
print (label_train_list.shape)

xyz_test_list=np.asarray(xyz_test_list)
label_test_list=to_categorical(np.asarray(label_test_list))

#xyz_train_list=normalize(xyz_train_list)
#xyz_test_list=normalize(xyz_test_list)
#xyz_train_list=normalize_2(xyz_train_list)
#xyz_test_list=normalize_2(xyz_test_list)
#print (xyz_train_list.shape)
#Activate cuda environment
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto(allow_soft_placement = True)
tf.GPUOptions(per_process_gpu_memory_fraction = 0.99)
config.gpu_options.allow_growth = True
sess= tf.Session(config = config)

#(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
#X_train=X_train.reshape(len(X_train),28,28,1)
#Y_train= to_categorical(Y_train)
#X_test=X_test.reshape(len(X_test),28,28,1)
#Y_test= to_categorical(Y_test)
#print (Y_train.shape)
#print (xyz_train_list.shape)
#print (label_train_list.shape)
model = createModel(xyz_train_list)
model.summary()
history=trainModel(model,xyz_train_list,label_train_list,xyz_test_list,label_test_list,epoch=50)


#for i in range(n_train):
#    print (training_file_list[i],label_train_list[i])



#print (label)

#print (dat.shape)
#print (data)

train_loss = history.history["loss"]
val_loss = history.history["val_loss"]
train_acc= history.history["acc"]
val_acc=history.history["val_acc"]

with open (sys.argv[0].replace(".py","")+"_train_loss.txt","w") as f:
    for loss in train_loss:
        f.write(str(loss)+"\n")

with open (sys.argv[0].replace(".py","")+"_val_loss.txt","w") as f:
    for loss in val_loss:
        f.write(str(loss)+"\n")

with open (sys.argv[0].replace(".py","")+"_train_acc.txt","w") as f:
    for acc in train_acc:
        f.write(str(acc)+"\n")

with open (sys.argv[0].replace(".py","")+"_val_acc.txt","w") as f:
    for acc in val_acc:
        f.write(str(acc)+"\n")