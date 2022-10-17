import copy
#https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
# https://github.com/horovod/horovod/blob/1b24d0f23d4a9965341851d935ebc2eff8bf886b/examples/adasum/adasum_small_model.py#L38
# https://github.com/horovod/horovod/blob/master/examples/pytorch/pytorch_imagenet_resnet50.py
import torch
# (442)
import torch.utils.data as data
import torch.nn as nn
import torchvision.transforms as transforms

import math
import os
import sys
import numpy as np

# READ ALL THESE FROM A FILE
import rough_copy
import torch.optim as optim

project_root = os.path.dirname(os.path.abspath(sys.argv[0]))
#feature_path = '/gpfs/alpine/proj-shared/bif135/raj/hetero_data/features/'
#label_path ='/gpfs/alpine/proj-shared/bif135/raj/hetero_data/labels/'
#feature_path = "/gpfs/alpine/proj-shared/bif135/raj/hetero_data/CASP/monomer_trRosetta_bfd/"
#feature_path = "/gpfs/alpine/proj-shared/bif135/raj/hetero_data/CASP/glinter_npz/"
feature_path= "/gpfs/alpine/proj-shared/bif135/raj/hetero_data/CASP/new_extracted_trRoss/"
#feature_path = '/gpfs/alpine/proj-shared/bif135/raj/hetero_data/CASP/trRossetta_monomer/'
label_path = "/gpfs/alpine/proj-shared/bif135/raj/hetero_data/CASP/Y-8A/"
val_file = "/gpfs/alpine/proj-shared/bif135/raj/hetero_data/CASP/casp_list.txt"
#feature_path = '/gpfs/alpine/proj-shared/bif132/farhan/data/HETERO_STD/trRosetta_monomer/'
#label_path = "/gpfs/alpine/proj-shared/bif132/farhan/data/HETERO_STD/Y-8A_including_transpose/"
#val_file = "../dataset/monomer_no_aug/monomer_het_val_list.txt"
#intra_cmap = "/gpfs/alpine/proj-shared/bif132/farhan/data/HETERO_STD/het_intra_cmap/"


#feature_path="/gpfs/alpine/proj-shared/bif132/farhan/data/HETERO_STD/selected_hetero/selected_trRosetta_joined/"
#label_path = "/gpfs/alpine/proj-shared/bif132/farhan/data/HETERO_STD/selected_hetero/Y-8A/"

#val_file="/gpfs/alpine/proj-shared/bif135/raj/hetero_data/latest_pdb.txt" 







#feature_path = '/gpfs/alpine/proj-shared/bif132/farhan/data/HETERO_STD/new_extracted_trRoss/'
#label_path = "/gpfs/alpine/proj-shared/bif132/farhan/data/HETERO_STD/Y-8A_including_transpose/"
#val_file = "/gpfs/alpine/proj-shared/bif135/raj/Hetero/monomerhet/het_pytorch_aug/dataset/cdhit_no_aug_monomer/aug/cdhit_aug_test_list.txt"





#val_file = "../dataset/initial/het_std_val_list.txt"

intra_cmap = "/gpfs/alpine/proj-shared/bif135/raj/hetero_data/CASP/intra_cmap/"


weighth_list =sys.argv[1]
output_folder =sys.argv[2]

if not os.path.exists(output_folder):
    os.system("mkdir -p "+ output_folder)

MAX_LENGTH = 500
EPOCH =100
FEATURES = 442
RESNET_DEPTH = 36
err_list = []


BATCH_SIZE = 1

device = torch.cuda.device(0)
torch.cuda.get_device_name(0)






def check_single_exists(_file):
    if os.path.exists(_file):
        return True
    else:
        print("missing "+str(_file))
        return False



def check_path_exists(_feat, _labels):
    if check_single_exists(_feat) and check_single_exists(_labels) :
        return True
    else:
        return False






def filter_files(_feat_path, _label_path, file_list):

    _feat_files = []
    _labels = []
    for val in file_list:
        feat_file_name = _feat_path +  str(val) + ".npz"
        label_name =  val.replace("Y-","").split("_")
        name_1 = label_name[0]
        name_2 = label_name[1].replace(".txt","")
#        labels_path_file_name = _label_path  +"Y-"+ str(name_2)+"_"+str(name_1) + ".txt"
        labels_path_file_name = _label_path  +"Y-"+ str(name_1)+"_"+str(name_2) + ".txt"

        if check_path_exists(_feat=feat_file_name,_labels=labels_path_file_name):
            _feat_files.append(feat_file_name)
            _labels.append(labels_path_file_name)

    return sorted(_feat_files), sorted(_labels)




def append_file_information(_filename, _info, _epoch, _type):
    exists_status = 0
    if os.path.exists(_filename):
        exists_status = 1

    if _type == "loss":
        with open(_filename, "a+") as f:
            f.write(str(_epoch) + " " + str(_info[0]) + "\n")
    else:
        if exists_status == 1:
            with open(_filename, "a+") as f:
                f.write(str(_epoch) + "\t\t\t" + str(_info[0]) + "\t\t\t" + str(_info[1]) + "\t\t\t" + str(
                    _info[2]) + "\t\t\t" + str(_info[3]) + "\t\t\t" + str(_info[4]) + "\t\t\t" + str(
                    _info[5]) + "\t\t\t" + str(_info[6]) + "\t\t\t" + str(_info[7]) + "\t\t\t" + str(
                    _info[8]) + "\t\t\t" + str(_info[9]) + "\n")
        else:
            with open(_filename, "a+") as f:
                f.write("Epoch\tSample_Size\tPrec-T5\tPrec-T10\tPrec-T20\tPrec-T30\tPrec-T50\tL/30\tL/20\tL/10\tL/5\n")

                f.write(str(_epoch) + "\t\t\t" + str(len(_info)) + "\t\t\t" + str(_info[1]) + "\t\t\t" + str(
                    _info[2]) + "\t\t\t" + str(_info[3]) + "\t\t\t" + str(_info[4]) + "\t\t\t" + str(
                    _info[5]) + "\t\t\t" + str(_info[6]) + "\t\t\t" + str(_info[7]) + "\t\t\t" + str(
                    _info[8]) + "\t\t\t" + str(_info[9]) + "\n")


def calculateEvaluationStats(_pred_cmap, _true_cmap, _L):
    prec_T5, prec_T10, prec_T20, prec_T30, prec_T50, prec_L30, prec_L20, prec_L10, prec_L5, con_num = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    pred_cmap = copy.deepcopy(_pred_cmap).squeeze()
    true_cmap = copy.deepcopy(_true_cmap).squeeze()
    L = _L

    if true_cmap.shape != pred_cmap.shape: print("True and predicted contact maps do not match in shape!")
    print("&&" * 30)
    max_Top = int((L / 5) + 0.5)
    if 50 > max_Top: max_Top = 50
    for i in range(1, max_Top + 1):
        (x, y) = np.unravel_index(np.argmax(pred_cmap, axis=None), pred_cmap.shape)
        pred_cmap[x][y] = 0
        if true_cmap[x][y] == 1:
            con_num += 1
        if i == 5:
            prec_T5 = con_num * 20
            if prec_T5 > 100: prec_T5 = 100
            print("L=", L, "Val=", 5, "Con_num=", con_num)
        if i == 10:
            prec_T10 = con_num * 10
            if prec_T10 > 100: prec_T10 = 100
            print("L=", L, "Val=", 10, "Con_num=", con_num)
        if i == 20:
            prec_T20 = con_num * 5
            if prec_T20 > 100: prec_T20 = 100
            print("L=", L, "Val=", 20, "Con_num=", con_num)
        if i == 30:
            prec_T30 = con_num * 100 / 30
            if prec_T30 > 100: prec_T30 = 100
            print("L=", L, "Val=", 30, "Con_num=", con_num)
        if i == 50:
            prec_T50 = con_num * 2
            if prec_T50 > 100: prec_T50 = 100
            print("L=", L, "Val=", 50, "Con_num=", con_num)
        if i == int((L / 30) + 0.5):
            prec_L30 = con_num * 100 / i
            if prec_L30 > 100: prec_L30 = 100
            print("L=", L, "Val=", i, "Con_num=", con_num)
        if i == int((L / 20) + 0.5):
            prec_L20 = con_num * 100 / i
            if prec_L20 > 100: prec_L20 = 100
            print("L=", L, "Val=", i, "Con_num=", con_num)
        if i == int((L / 10) + 0.5):
            prec_L10 = con_num * 100 / i
            if prec_L10 > 100: prec_L10 = 100
            print("L=", L, "Val=", i, "Con_num=", con_num)
        if i == int((L / 5) + 0.5):
            prec_L5 = con_num * 100 / i
            if prec_L5 > 100: prec_L5 = 100
            print("L=", L, "Val=", i, "Con_num=", con_num)
    del pred_cmap, true_cmap
    return prec_T5, prec_T10, prec_T20, prec_T30, prec_T50, prec_L30, prec_L20, prec_L10, prec_L5



def text_file_reader(_file):
    err_list =[] 
    file = open(_file, "r")
    output_array = []
    if file.mode == 'r':
        output_array = file.read().splitlines()
    file.close()
    for error in err_list:
        if error.strip() in output_array:
            output_array.remove(error.strip())

    return output_array





class my_dataset(data.Dataset):
    def initialize(self, _feat_path,  _label_path, _file_list, _max_len):
        file_list = _file_list
        self.features,   self.labels = filter_files(_feat_path, _label_path,   file_list)
        self.size = len(self.features)
        self.size_labels = len(self.labels)
        self.MaxLen = _max_len
        self.name = ""
        self.len_a = 0
        self.len_b = 0
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, index):
        feature_path = self.features[index % self.size]
        path_labels = self.labels[index % self.size_labels]
        print(os.path.basename(path_labels))

        content = np.load(feature_path, allow_pickle=True)
        tr_feature = content.f.arr_0.squeeze()
        
        ground_truth = np.loadtxt(path_labels)
        lab_a_len,lab_b_len= ground_truth.shape
        self.name = os.path.basename(path_labels)


        len_a,len_b,fetures = tr_feature.shape
        self.len_a = len_a
        self.len_b = len_b
       # print(len_a,len_b)
        #print(lab_a_len,lab_b_len)
        
        if len_a != lab_a_len or len_b!=lab_b_len:
            print(feature_path)
            print(path_labels)
            print(len_a, len_b)
            print(lab_a_len, lab_b_len)
            print("mistmatch"+str(os.path.basename(feature_path))+","+str(len_a)+","+str(len_b)+","+str(lab_a_len)+","+str(lab_b_len))



      # PADDING FEATURES
        final_feature = np.zeros((self.MaxLen, self.MaxLen, FEATURES))

        for val in range(0, len_a):
            final_feature[val, 0:len_b, 0:FEATURES] = tr_feature[val]


        dist = self.transform(final_feature)
        padded_labels = np.zeros((self.MaxLen, self.MaxLen))
        for val in range(0, len_a):
            padded_labels[val][0:len_b] = ground_truth[val]


    
    

        return {'feat': dist, 'ground_truth': padded_labels, 'len_a': self.len_a, 'len_b': self.len_b,
                'name': self.name}
    def __len__(self):
        return self.size





model = rough_copy.ResNet_custom(img_channel=FEATURES, num_classes=MAX_LENGTH * MAX_LENGTH, _depth=RESNET_DEPTH)
model.cuda()

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0)


Current_EPOCH = 0
weights =weighth_list

if len(weights) > 0:
#    resume_from_epoch = hvd.broadcast(torch.tensor(weights), root_rank=0).item()
    last_weight =  weights
    print("Weight loading "+str(weights))
    checkpoint = torch.load(last_weight)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    Current_EPOCH = epoch


#print(last_weight)
nThreads=6

print("VALIDATION DATA LOADER AREA ")
val_file_list = text_file_reader(val_file)
print(len(val_file_list))
val_dataset = my_dataset()
val_dataset.initialize(_feat_path=feature_path, _label_path=label_path, _file_list=val_file_list,_max_len=MAX_LENGTH)

#val_dataset.initialize(_feat_path=feature_path, _label_path=label_path,_file_list=val_file_list, _max_len=MAX_LENGTH)
#
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=nThreads)
print(len(val_dataloader))
print("MODELLOADER AREA ")
def Average(lst):
    return round(sum(lst) / len(lst), 5)

last_epoch =1



print("HVD  AREA ")

with torch.no_grad():
    list_acc_T5, list_acc_T10, list_acc_l30, list_acc_l20, list_acc_l10, list_acc_l5, list_acc_T20, list_acc_T30, list_acc_T50 = [], [], [], [], [],[], [], [], []
    for i, data in enumerate(val_dataloader):
        print(data['name'])

        features = data['feat'].float().cuda()
        label = data['ground_truth'].float().cuda()
        real_sequence_length_a = data['len_a']
        real_sequence_length_b = data['len_b']
        model.eval()
#        print(data['name'])
        output_val = model(features)
        output_final = torch.squeeze(output_val, -1).detach().cpu().clone().numpy().squeeze()
        print(output_final)
        loss = criterion(torch.squeeze(output_val), torch.squeeze(label))
        print(loss)
       # running_validation_loss += loss.item()
            #            val_loss.append(loss.item())
     #       val_loss.update(loss)
        mini_batch_size = len(label)
        np_label = torch.squeeze(label, -1).detach().cpu().clone().numpy().squeeze()
          #  i_len = math.floor(math.sqrt(real_sequence_length_b*real_sequence_length_a))

        for counter in range(0, mini_batch_size):
            if mini_batch_size == 1:
                #shape_label = shape_label_array
                padded_predicted_label = output_final
                padded_true_label_label = np_label
            else:
                shape_label = shape_label_array[counter]
                padded_predicted_label = output_final[counter]
                padded_true_label_label = np_label[counter]
            unpadded_predicted_label = np.zeros((real_sequence_length_a, real_sequence_length_b))
            unpadded_true_label = np.zeros((real_sequence_length_a, real_sequence_length_b))
            for val in range(0, real_sequence_length_a):
                unpadded_predicted_label[val] = padded_predicted_label[val][0:real_sequence_length_b]
                unpadded_true_label[val] = padded_true_label_label[val][0:real_sequence_length_b]
            i_len = math.floor(math.sqrt(real_sequence_length_b * real_sequence_length_a)) 
            np.savetxt( output_folder+"/"+str(data['name'])+".cmap" ,unpadded_predicted_label)

            (prec_T5, prec_T10, prec_T20, prec_T30, prec_T50, prec_L30, prec_L20, prec_L10, prec_L5) = calculateEvaluationStats(_pred_cmap=unpadded_predicted_label, _true_cmap=unpadded_true_label, _L=i_len)
            list_acc_T5.append(prec_T5)
            list_acc_T10.append(prec_T10)
            list_acc_T20.append(prec_T20)
            list_acc_T30.append(prec_T30)
            list_acc_T50.append(prec_T50)
            list_acc_l30.append(prec_L30)
            list_acc_l20.append(prec_L20)
            list_acc_l10.append(prec_L10)
            list_acc_l5.append(prec_L5)

   # val_acc_info = [len(list_acc_T5), Average(list_acc_T5), Average(list_acc_T10), Average(list_acc_T20),  Average(list_acc_T30), Average(list_acc_T50) , Average(list_acc_l30), Average(list_acc_l20), Average(list_acc_l10), Average(list_acc_l5)]
   # append_file_information(_filename=LOG_DIR + "/validation_acc" + str(Current_EPOCH) + "txt",   _epoch=Current_EPOCH,                                _info=val_acc_info,                                _type="Accuracy")
