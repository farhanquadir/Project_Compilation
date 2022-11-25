#!/usr/bin/env python
# coding: utf-8

import torch.nn as nn                                                                      
from torchvision import models
import os,sys
import torch
import numpy as np
from torchvision import datasets
import torchvision
import matplotlib.pyplot as plt
import torch.utils.data as tech 
from tqdm import tqdm
import seaborn as sn  
import pandas as pd
from torchsummary import summary

# lets name oue neural network "CNN"
class CNN(nn.Module):
    
    # this is our CNN initilization function     
    def __init__(self, size, num_classes):
        super(CNN, self).__init__()
        self.extract = nn.Sequential( # lets make a 2D convolution layer
                                      nn.Conv2d( in_channels = size, out_channels = 10, 
                                                 kernel_size = (28,28), stride = 1, padding = 0), 
                                      nn.ReLU(inplace = True)#,nn.Sigmoid()
                                      #nn.MaxPool2d(2),                                                                
                                      #nn.Dropout(0.1),
                                      # ----------------------------------------------------------- 
                                      # now, lets make another layer of convolution, pooling, and drop out
                                      #nn.Conv2d( in_channels = 2, out_channels = 4, 
                                                 #kernel_size = 3, stride = 1, padding = 1),
                                                 # in_channels here needs to match out_channels above
                                                 # lets use 5 filters 
                                      #nn.ReLU(inplace = True),
                                      #nn.MaxPool2d(2),
                                      #nn.Dropout(0.1), 
                                    )

        # ok, now we are going to make a simple MLP classifier on the end of our above features
        self.decimate = nn.Sequential( nn.Linear(10,2), nn.Sigmoid())#, nn.Linear(10,10))
        """
        self.decimate = nn.Sequential( nn.Linear(10*(11*11), 12),  
                                            # take our 10 filters whose response fields are 11x11 to 12 neurons
                                       nn.ReLU(inplace = True), # run a nonlinearity
                                       nn.Dropout(0.2), # some drop out
                                       nn.Linear(12, num_classes) ) # map the 32 down to our number of output classes
 """
    #----------------------------
    # Model: Invoke Forward Pass
    #----------------------------

    def forward(self, x):

        features = self.extract(x) # easy, pass input (x) to our "feature extraction" above
        features = features.view(features.size()[0], -1) # now, flatten 7x7x4 matrix to 1D array of 7*7*4 size
        #myresult = features
        myresult=self.decimate(features) # pass that to our MLP classifier, and done!!!

        return myresult


# Next, lets load our training data set for MNIST


# nice built in functions for common data sets 
#  go read https://pytorch.org/docs/stable/torchvision/datasets.html
train = datasets.MNIST( root = '../', # where to download data set to
                       train = True, # If True, creates dataset from training.pt, otherwise from test.pt
                       transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor()]), # convert data into tensor vs PIL image 
                       download = True)
c1 = 3
c2 = 8
trainidx = torch.LongTensor(train.targets) == c1
trainidx += torch.LongTensor(train.targets) == c2

#print (trainidx)
#print(len(trainidx ==1))
#print(len(trainidx ==1))
#print (trainidx)
#print (trainidx[0])
#sys.exit()
#testidx = torch.LongTensor(test.targets) == c1
#testidx += torch.LongTensor(test.targets) == c2
#train = torch.utils.data.dataset.Subset(train, np.where(trainidx==1)[0])

train.targets= train.targets[trainidx]
train.data = train.data[trainidx]
#test = torch.utils.data.dataset.Subset(test, np.where(testidx==1)[0])
print("Train_shape=",train.data.shape)
print("Train_shape=",train.targets.shape)

test = datasets.MNIST( root = '../', 
                       train = False, 
                       transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor()]), # convert data into tensor vs PIL image                       
                       download = True)
testidx = torch.LongTensor(test.targets) == c1
testidx += torch.LongTensor(test.targets) == c2
test.targets= test.targets[testidx]
test.data = test.data[testidx]

#sys.exit()

"""
train = datasets.FashionMNIST( root = './', # where to download data set to
                       train = True, # If True, creates dataset from training.pt, otherwise from test.pt
                       transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor()]), # convert data into tensor vs PIL image 
                       download = True)
"""
print("[num of images, image x size, image y size]")
#print(train.dataset[0][0].shape)
#sys.exit()
print("what type of data is it?")
#print(type(train.data[0]))

print("what is min and max values?")
#print(torch.max(train.dataset[0][]))
#print(torch.min(train.data[0]))

# lets plot it
#import matplotlib.pyplot as plt

#plt.imshow(train.data[0])
#print (train.data[0])
#plt.imshow(train.data[0])
#print (train.data[0].shape)
#print (train.target[0].shape)
#print (train.targets.shape)
print ("DONE")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
if (torch.cuda.is_available()):
    print ("GPU Found: Selected device: Cuda:0")
    #torch.cuda.set_device(0)
    #device=torch.device("cuda:0")
    #model.to(device)
else:
    print ("GPU Not Available: Selected Device: Cpu")


# If you want to work with validation or test data, follow
# 
#     valid = datasets.MNIST( root = './', train = False, download = True)
#     test = datasets.MNIST( root = './', train = False, download = True)

# Lets now make a data loader object to hold onto our data that we can use for batch processing and stuff



# how big of batches do you guys/gals want?
batch_size = 1

# our data loader that we will use to manage our data
train_ld = tech.DataLoader(dataset = train, shuffle = True, batch_size = batch_size)       
test = tech.DataLoader(dataset = test, shuffle = False, batch_size = 1) 
#print(train_ld.dataset.targets[0])
print("Max target value is = ",max(train_ld.dataset.targets))
print("Min target value is = ",min(train_ld.dataset.targets))

#sys.exit()

# Again, you can work with validation and test data as well
#     
#     valid = tech.DataLoader(dataset = valid, shuffle = False, batch_size = batch_size)      
#     test = tech.DataLoader(dataset = test, shuffle = False, batch_size = 1) 

# Lets create an instance of our CNN


input_size = 1 # just 1 band for MNIST
num_classes = 2 # we have 10 classes in MNIST
model = CNN(input_size, num_classes).to(device) 

#device = torch.device("cuda")
#model.to(device)


#print (model)
print(summary(model,(1,28,28)))
#sys.exit()

# Now, pick optimization algorithm and error function


learning_rate = 0.0001
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
#costfx = torch.nn.CrossEntropyLoss()
costfx = torch.nn.BCELoss()
#costfx = torch.nn.MSELoss(reduction="sum")

# Training time!
print ("total number of training data: ",len(train_ld.dataset.data))
print ("total number of training labels: ",len(train_ld.dataset.targets))
len_train=len(train_ld.dataset.data)
len_test=len(test.dataset.data)
#sys.exit()

#from tqdm import tqdm#_notebook as tqdm
label_dict={}
label_dict[3]=0
label_dict[8]=1

num_epochs = 100
print ("Total Epochs=",num_epochs)
total_error=[] 
total_val_error=[]          
val_accuracy=[]               
train_accuracy=[]
for epoch in range(num_epochs): # how many epochs? 
    
    epoch_loss = [] # keep track of our loss?
    print ("Running epoch "+str(epoch+1)+"/"+str(num_epochs)+"...")
    train_acc_count=0
    for batch_id, train_params in enumerate(tqdm(train_ld)):  # lets grab a bunch of mini-batches from our training data set
        
        # samples are our images, labels are their class labels
        samples, labels = train_params[0].to(device),train_params[1].to(device)
        ll=torch.zeros(1,2,device=device)
        #print("ll.shape=",ll.shape)
        # we need to convert these into tensors
#        samples = samples.type('torch.FloatTensor') 
#        labels = labels.type('torch.LongTensor')  
        labs=int(labels.item())
        #print("Labs=",labels.item(),type(labels.item()))
        if (int(labels.item())==c1):
            ll.data[0][0]=1#labels.item()
            labels.data[0]=0
        else:
            ll.data[0][1]=1#labels.item()
            labels.data[0]=1
        #ll.data[0][labels.item()]=labels.item()
        
        #ll.data[0][0]=labels.item()
        labels=ll
        #print (samples[0])
        #print (labels[0])
        #print ("Label_shape=",labels.shape)
        #print ("Labels=",labels)
        #plt.imshow(samples[0][0])
        #sys.exit()

        # lets predict (forward pass)
        prediction = model(samples)
        
        #label_acc = int(labels.numpy())
        #prediction_acc = int(prediction.item())
        #if (label_acc==prediction_acc): train_acc_count+=1
        #print ("Prediction.shape=",prediction.shape)
        #print ("Labels.shape=",labels.shape)
        #sys.exit()
        # evaluate our error
        #print (prediction)
        #print(torch.max(prediction).data)
#        predict=torch.max(prediction).data.type("torch.FloatTensor")
        #predict=max(prediction).data#.type("torch.FloatTensor")
        #print("Predict=",predict)
#        sys.exit()
        loss = costfx(prediction, labels)
        #sys.exit()
        # keep track of that loss
        epoch_loss.append(loss.item())
        # zero our gradients
        optimizer.zero_grad()  
        # calc our gradients
        
        loss.backward()     
        
        # do our update
        optimizer.step()
        #break
        ################ Add accuracy code here!
        label_acc = int(label_dict[labs])
        prediction_acc = int(prediction.argmax())
        #print ("ASDFADSF")
        #print (labels,prediction)
        #sys.exit()
        if (label_acc==prediction_acc): 
            train_acc_count+=1
#            torch_pred=torch.Tensor(prediction)
            #print (torch_pred)
            #print (torch_pred.data[0])
            #print (int(torch_pred.data[0][0]+0.5))
            #print (int(torch_pred.data[0][1]+0.5))
#            s="("+str(torch_pred.data[0][0])+", "+str(torch_pred.data[0][1])+")"
            #print (s)
            #print ("echo "+str(label_acc)+" "+str(prediction_acc)+" "+str(labs)+" "+s+" >> train_track.txt")
            #sys.exit()
#            os.system("echo '"+str(label_acc)+" "+str(prediction_acc)+" "+str(labs)+" "+s+"' >> train_track.txt")
    #sys.exit()
    # keep track of loss over our batches
    #epoch_loss = sum(epoch_loss)/len(epoch_loss)  
    #print ("Here!")
    total_error.append(sum(epoch_loss)/len(epoch_loss))
    print ("Training loss = ",sum(epoch_loss)/len(epoch_loss))
    train_accuracy.append(train_acc_count/len_train)
    print("Training accuracy=",train_acc_count/len_train)
    
    epoch_loss=[]
    print ("Running validation for test data...")
    print ("Total number of validation data: ",len(test.dataset.data))
    print ("Total number of validation labels: ",len(test.dataset.targets))
    val_acc_count=0
    for sample, labels in tqdm(test):
        # what is its label?
        #label = int(label.numpy())
        #print("Real label is")
        #print(label)
        # convert the sample (image) to a tensor for PyTorch
        ll=torch.zeros(1,2,device=device)
        sample = sample.to(device)#sample.type('torch.FloatTensor')
        labels = labels.to(device)#labels.type('torch.LongTensor')  
        labs=int(labels.item())
        #label_acc = int(labels.numpy())
        
        if (int(labels.item())==c1):
            ll.data[0][0]=1#labels.item()
            labels.data[0]=0
        else:
            ll.data[0][1]=1#labels.item()
            labels.data[0]=1
        labels=ll
        
        prediction = model(sample) 
        prediction_acc = int(prediction.argmax())
        label_acc = int(label_dict[labs])
        
        if (label_acc==prediction_acc): 
            val_acc_count+=1
            #torch_pred=torch.Tensor(prediction)
            #s="("+str(torch_pred.data[0][0])+", "+str(torch_pred.data[0][1])+")"
            #os.system("echo '"+str(label_acc)+" "+str(prediction_acc)+" "+str(labs)+" "+s+"' >> val_track.txt")
        #ll.data[0][labels.item()]=labels.item()
        #ll.data[0][0]=labels.item()
        
        
        # do forward pass (i.e., prediction)
        
        # take the largest output and return integer of which it was (make a classification decision)
        #prediction = int(torch.argmax(prediction).numpy())
        loss=costfx(prediction,labels)
        epoch_loss.append(loss.item())
        # what was our prediction?
        #print(prediction)
        ################ Add accuracy code here!
        
    total_val_error.append(sum(epoch_loss)/len(epoch_loss))
    print ("Validation loss = ",sum(epoch_loss)/len(epoch_loss))
    val_accuracy.append(val_acc_count/len_test)
    print("Validation accuracy=",val_acc_count/len_test)


with open ("./"+sys.argv[0].replace(".py","_train_error.txt"),"w") as f:
    for err in total_error:
        f.write(str(err)+"\n")

with open ("./"+sys.argv[0].replace(".py","_val_error.txt"),"w") as f:
    for err in total_val_error:
        f.write(str(err)+"\n")

with open ("./"+sys.argv[0].replace(".py","_train_acc.txt"),"w") as f:
    for acc in train_accuracy:
        f.write(str(acc)+"\n")

with open ("./"+sys.argv[0].replace(".py","_val_acc.txt"),"w") as f:
    for acc in val_accuracy:
        f.write(str(acc)+"\n")


# Save the model

torch.save(model, './'+sys.argv[0].replace(".py","_model.pt")) 



# Render the filters
wghts=len(model.extract[0].weight)
print ("Filters= ",wghts)
for i in range(wghts):
    plt.figure()
    #plt.imsave("./"+sys.argv[0].replace(".py","_weight_"+str(i)+".png"), np.squeeze( model.extract[0].weight[i,:,:,:].detach().numpy()))
    plt.imshow(np.squeeze( model.extract[0].weight[i,:,:,:].cpu().detach().numpy()))
    plt.savefig("./"+sys.argv[0].replace(".py","_weight_"+str(i)+".png"))
    np.savetxt("./"+sys.argv[0].replace(".py","_weight_"+str(i)+".txt"),np.squeeze( model.extract[0].weight[i,:,:,:].cpu().detach().numpy()))
    #print(model.extract[0].weight.shape)
    plt.close()


# Next, lets scrub all the junk in our net that was needed at training time, like dropout


model = model.eval()


# Lets do resub, load back up a data point and see how we did ...

# how did we do...
count_dict={}
for i in range(10):
    count_dict[i]=0
    
ConfusionMatrix = np.zeros((2,2),dtype=int)
for sample, label in test:
    # what is its label?
    #labs=label.copy()
    labels=label
    label = int(label.item())
    count_dict[label]+=1
    labs = label
    #print("Real label is")
    #print(label)
    # convert the sample (image) to a tensor for PyTorch
    sample = sample.to(device)#sample.type('torch.FloatTensor')
    # do forward pass (i.e., prediction)
    prediction = model(sample) 
    #print ("Actual_pred=",prediction.item())
    # take the largest output and return integer of which it was (make a classification decision)
    #print (prediction)
    #print (int(torch.argmax(prediction).numpy()))
    #print (torch.argmax(prediction).numpy())
    #print (label)
    #prediction = prediction.numpy()[int(torch.argmax(prediction).numpy())]
    #print (prediction)
    
    #val_acc_count+=1
    #torch_pred=torch.Tensor(prediction)
    #s="("+str(torch_pred.data[0][0])+", "+str(torch_pred.data[0][1])+")"
    prediction_acc = int(prediction.argmax())
    #label_acc = int(torch.argmax(labels).numpy())
    #os.system("echo '"+str(label)+" "+str(prediction_acc)+" "+str(labs)+" "+s+"' >> conf_val_track.txt")
    #label = int(torch.argmax(labels).numpy())
    prediction = prediction_acc
    #print (label,labels,label_acc,prediction_acc)
    #sys.exit()
    #print ("Prediction=",prediction)
    # what was our prediction?
    #print(prediction)
    if (label==c1): 
        label=0
    else:
        label=1
    #if (prediction==0):
    #    prediction=0
    #else:
    #    prediction=1
    #print ("(",label,prediction,")")
    #print (label,prediction,labs)
    ConfusionMatrix[label,prediction] += 1
    
df_cm = pd.DataFrame(np.asarray(ConfusionMatrix,dtype=int), index = [i for i in ["3","8"]],
                  columns = [i for i in ["3","8"]])

print (df_cm.to_string())
plt.figure(figsize = (10,7))
#plt.figure()
sn.set()
sns_plot=sn.heatmap(df_cm, fmt="d",annot=True)
fig=sns_plot.get_figure()
fig.savefig("./"+sys.argv[0].replace(".py","_confusion_mat.png"))
#plt.show()
plt.close()
print (count_dict)
for key,val in count_dict.items():
    print (key,":",val)
#print (count_dict)

