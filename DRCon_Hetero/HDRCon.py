import copy
import torch
# (442)
import torch.utils.data as data
import torch.nn as nn
import torchvision.transforms as transforms
import math
import os
import sys
import numpy as np
import rough_copy
import torch.optim as optim

project_root = os.path.dirname(os.path.abspath(sys.argv[0]))
feature_path= sys.argv[1]


model_name =sys.argv[2]
output_folder =sys.argv[3]

device = torch.device("cpu")
MAX_LENGTH = 500
EPOCH =100
FEATURES = 442
RESNET_DEPTH = 36
err_list = []


BATCH_SIZE = 1



def check_single_exists(_file):
    if os.path.exists(_file):
        return True
    else:
        print("missing "+str(_file))
        return False




def filter_files(_feat_path):

    _feat_files = []

    if check_single_exists(_feat_path):
        print("file found "+str(_feat_path))
        _feat_files.append(_feat_path)
        return _feat_files
    else:
        print("file found "+str(_feat_path))
        return False
  



class my_dataset(data.Dataset):
    def initialize(self, _feat_path, _max_len):
 
        self.features = filter_files(_feat_path)
        self.size = len(self.features)
        self.MaxLen = _max_len
        self.name = ""
        self.len_a = 0
        self.len_b = 0
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, index):
        feature_path = self.features[index % self.size]
 

        content = np.load(feature_path, allow_pickle=True)
        tr_feature = content.f.arr_0.squeeze()
        self.name = os.path.basename(feature_path)
        len_a,len_b,fetures = tr_feature.shape
        self.len_a = len_a
        self.len_b = len_b
       # print(len_a,len_b)
        #print(lab_a_len,lab_b_len)
        



      # PADDING FEATURES          
        max_len_a = max(len_a,500)
        max_len_b = max(len_b,500)
        final_feature = np.zeros((max_len_a, max_len_b, FEATURES))
        # final_feature = np.zeros((self.MaxLen, self.MaxLen, FEATURES))
        for val in range(0, len_a):
            final_feature[val, 0:len_b, 0:FEATURES] = tr_feature[val]
        

        dist = self.transform(final_feature)
        return {'feat': dist, 'len_a': self.len_a, 'len_b': self.len_b,'name': self.name}

    def __len__(self):
        return self.size





model = rough_copy.ResNet_custom(img_channel=FEATURES, num_classes=MAX_LENGTH * MAX_LENGTH, _depth=RESNET_DEPTH)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0)


Current_EPOCH = 0
weights =model_name

if os.path.exists(model_name):
    print("model found")
 
    # checkpoint = torch.load(model_name).to(device)
    checkpoint = torch.load(model_name,map_location ='cpu')
    model.load_state_dict(checkpoint['model']) 
    print("model loaded ")
else:
    print("model not found")

#print(last_weight)
nThreads=6
  
val_dataset = my_dataset()
val_dataset.initialize(_feat_path=feature_path,_max_len=MAX_LENGTH)
  
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, num_workers=nThreads)

 

with torch.no_grad():
    for i, data in enumerate(val_dataloader):
        print(data['name'])

        features = data['feat'].to(device).float() 
     
        real_sequence_length_a = data['len_a']
        real_sequence_length_b = data['len_b']
        model.eval()

        output_val = model(features)
        output_final = torch.squeeze(output_val, -1).detach().cpu().clone().numpy().squeeze()
         
        
       
 
 
          #  i_len = math.floor(math.sqrt(real_sequence_length_b*real_sequence_length_a))
        padded_predicted_label = output_final       
        unpadded_predicted_label = np.zeros((real_sequence_length_a, real_sequence_length_b))
        
        for val in range(0, real_sequence_length_a):
            unpadded_predicted_label[val] = padded_predicted_label[val][0:real_sequence_length_b]
            # unpadded_true_label[val] = padded_true_label_label[val][0:real_sequence_length_b]
        i_len = math.floor(math.sqrt(real_sequence_length_b * real_sequence_length_a)) 
        print(str(data['name']))
        np.savetxt( output_folder+"/"+str(data['name'][0]).replace(".npz","")+".cmap" ,unpadded_predicted_label)


   # val_acc_info = [len(list_acc_T5), Average(list_acc_T5), Average(list_acc_T10), Average(list_acc_T20),  Average(list_acc_T30), Average(list_acc_T50) , Average(list_acc_l30), Average(list_acc_l20), Average(list_acc_l10), Average(list_acc_l5)]
   # append_file_information(_filename=LOG_DIR + "/validation_acc" + str(Current_EPOCH) + "txt",   _epoch=Current_EPOCH,                                _info=val_acc_info,                                _type="Accuracy")
