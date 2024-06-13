import sys, getopt
import torch
from torch.utils import data
import numpy as np
import pandas as pd
from PIL import Image
import albumentations as A
import time
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import torch.utils.data
from sklearn import metrics 
import os
import argparse
from glob import glob
import json
import warnings
warnings.filterwarnings("ignore")
import pyspng
import random
from model import MoCo, simCLR
from data_augmentation import get_pipeline_augment
from dataloader import Dataset_instance_eval, Dataset_bag_multilabel
import torch.optim as optim
from loss_functions import ContrastiveLoss, NT_Xent, SimCLR_Loss
from utils import get_generator_instances, generate_list_instances

#from pytorch_pretrained_bert.modeling import BertModel

args = sys.argv[1:]

print("CUDA current device " + str(torch.cuda.current_device()))
print("CUDA devices available " + str(torch.cuda.device_count()))

#parser parameters
parser = argparse.ArgumentParser(description='Configurations to train models.')
parser.add_argument('-n', '--N_EXP', help='number of experiment',type=int, default=0)
parser.add_argument('-c', '--CNN', help='cnn_to_use',type=str, default='resnet34')
parser.add_argument('-m', '--MAG', help='magnification to select',type=str, default='10')
parser.add_argument('-a', '--ALGORITHM', help='algorithm to use (MoCo, simCLR)',type=str, default='simCLR')
parser.add_argument('-t', '--TISSUE', help='tissue to use (Colon, Celiac, Lung)',type=str, default='Colon')
parser.add_argument('-p', '--PATH_MODEL', help='path of the file including model weights',type=str, default='')
parser.add_argument('-i', '--FOLDER_DATA', help='folder where patches are stored',type=str, default='')
parser.add_argument('-l', '--LIST_IMAGES', help='path of the csv file including the WSI IDs',type=str, default='')
parser.add_argument('-k', '--keys', help='number of keys',type=int, default=32768)


args = parser.parse_args()

N_EXP = args.N_EXP
N_EXP_str = str(N_EXP)
CNN_TO_USE = args.CNN
MAGNIFICATION = args.MAG
MAGNIFICATION_str = str(MAGNIFICATION)
ALGORITHM = args.ALGORITHM
TISSUE = args.TISSUE
num_keys = args.keys

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

seed = N_EXP
torch.manual_seed(seed)
if torch.cuda.is_available():
	torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

print("PARAMETERS")
print("CNN used: " + str(CNN_TO_USE))
print("MAGNIFICATION: " + str(MAGNIFICATION))

#path model file
model_weights_filename = args.PATH_MODEL


#CSV LOADING
print("CSV LOADING ")

k = 10

instance_dir = args.FOLDER_DATA

csv_filename_k_folds = args.LIST_IMAGES


images = pd.read_csv(csv_filename_k_folds, sep=',', header=None).values[:,:6]



n_shuffle = np.random.randint(100, size=1)[0]
n_shuffle = random.randint(1,50)
#"""
for n in range(n_shuffle):
	np.random.shuffle(images)
#"""
#MODEL DEFINITION
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_dim = 512
hidden_dim = 128
output_dim = 128


try:
	model = torch.load(model_weights_filename).to(device)
	model.eval()
except:

	if (ALGORITHM=='simCLR'):
		model = simCLR(CNN_TO_USE, in_dim=input_dim, out_dim=output_dim, intermediate_dim=hidden_dim)
	elif (ALGORITHM=='MoCo'):
		momentum = 0.999
		TEMPERATURE = 0.07
		model = MoCo(CNN_TO_USE, K=num_keys, m=momentum, T=TEMPERATURE, in_dim=input_dim, intermediate_dim = hidden_dim, out_dim = output_dim)

	model.load_state_dict(torch.load(model_weights_filename), strict=False)
	model.eval()
	
model.to(device)
	
batch_size_bag = 1

params_train_bag = {'batch_size': batch_size_bag,
	  'shuffle': True}

validation_set_bag = Dataset_bag_multilabel(images[:,0], images[:,1:])
validation_generator_bag = data.DataLoader(validation_set_bag, **params_train_bag)

iterations = len(images)

tot_batches_training = iterations#int(len(train_dataset)/batch_size_bag)

dataloader_iterator = iter(validation_generator_bag)
batch_size_instance = 512

model.eval()
mode = 'eval'

for i in range(iterations):
	print('%d / %d ' % (i, tot_batches_training))
	try:
		ID, _ = next(dataloader_iterator)
	except StopIteration:
		dataloader_iterator = iter(validation_generator_bag)
		ID, _ = next(dataloader_iterator)
		#inputs: bags, labels: labels of the bags
		
	filename_wsi = ID[0]
	input_bag_wsi = generate_list_instances(instance_dir, filename_wsi)

	

	filename_features = instance_dir+'/'+filename_wsi+'/'+filename_wsi+'_features_'+ALGORITHM+'.npy'

	flag_b = os.path.isfile(filename_features)

	instances_filename_sample = generate_list_instances(instance_dir, filename_wsi)

	flag_c = True
	try:
		csv_instances = pd.read_csv(instances_filename_sample, sep=',', header=None).values
		n_elems = len(csv_instances)
		flag_c == True

	except Exception as e:
		print(e)
		flag_c = False

	print("[" + str(i) + "/" + str(len(images)) + "], " + "inputs_bag: " + str(filename_wsi)) 
	print(flag_c)
	print(flag_b)

	if (flag_b==False and flag_c == True):
		
		try:
			
			training_generator_instance = get_generator_instances(csv_instances, batch_size_instance)

			model.eval()

			features = []
			with torch.no_grad():
				for instances in training_generator_instance:
					instances = instances.to(device, non_blocking=True)

					# forward + backward + optimize
					if (ALGORITHM=='simCLR'):
						feats = model.base_encoder.conv_layers(instances)
					elif (ALGORITHM=='MoCo'):
						feats = model.encoder_q.conv_layers(instances)

					feats = feats.view(-1, input_dim)
					feats_np = feats.cpu().data.numpy()
					
					features = np.append(features,feats_np)
					
					del instances
				#del instances
			features_np = np.reshape(features,(n_elems,input_dim))

			torch.cuda.empty_cache()
			del features, feats

			with open(filename_features, 'wb') as f:
				np.save(f, features_np)
		
		except FileNotFoundError as e:
			pass
			
torch.cuda.empty_cache()