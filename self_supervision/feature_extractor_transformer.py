import sys, getopt
import torch
from torch.utils import data
import numpy as np
import pandas as pd
from PIL import Image
import albumentations as A
import time
import torch.nn.functional as F
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
import vision_transformer
import utils_transformer 
from functools import partial
from torchvision import datasets, transforms

#from pytorch_pretrained_bert.modeling import BertModel

args = sys.argv[1:]

print("CUDA current device " + str(torch.cuda.current_device()))
print("CUDA devices available " + str(torch.cuda.device_count()))

#parser parameters
parser = argparse.ArgumentParser(description='Configurations to train models.')
parser.add_argument('-n', '--N_EXP', help='number of experiment',type=int, default=0)
parser.add_argument('-c', '--CNN', help='cnn_to_use',type=str, default='resnet34')
parser.add_argument('-m', '--MAG', help='magnification to select',type=str, default='10')
parser.add_argument('-a', '--ALGORITHM', help='algorithm to use (MoCo, simCLR)',type=str, default='DINO')
parser.add_argument('-t', '--TISSUE', help='tissue to use (Colon, Celiac, Lung)',type=str, default='Colon')
parser.add_argument('-p', '--PATH_MODEL', help='path of the file including model weights',type=str, default='')
parser.add_argument('-i', '--FOLDER_DATA', help='folder where patches are stored',type=str, default='')
parser.add_argument('-l', '--LIST_IMAGES', help='path of the csv file including the WSI IDs',type=str, default='')

args = parser.parse_args()

N_EXP = args.N_EXP
N_EXP_str = str(N_EXP)
CNN_TO_USE = args.CNN
MAGNIFICATION = args.MAG
MAGNIFICATION_str = str(MAGNIFICATION)
ALGORITHM = args.ALGORITHM
TISSUE = args.TISSUE


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

MoCo_TO_USE = N_EXP_str

#models_path = '/PATH/TO/DINO/MODEL'
#path model file
model_weights_filename = args.PATH_MODEL


#CSV LOADING
print("CSV LOADING ")

instance_dir = args.FOLDER_DATA

csv_filename_k_folds = args.LIST_IMAGES


images = pd.read_csv(csv_filename_k_folds, sep=',', header=None).values[:,:6]



n_shuffle = np.random.randint(100, size=1)[0]
n_shuffle = random.randint(1,100)
#"""
for n in range(n_shuffle):
	np.random.shuffle(images)
#"""
#MODEL DEFINITION
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_dim = 512
hidden_dim = 512
output_dim = 128

patch_size = 16


try:
	print("load from weights")
	model = torch.load(model_weights_filename).to(device)
	model.eval()
except:

	print("load from scratch")
	model = vision_transformer.VisionTransformer(
		patch_size=patch_size, embed_dim=hidden_dim, depth=4, num_heads=8, mlp_ratio=4,
		qkv_bias=True, norm_layer=partial(torch.nn.LayerNorm, eps=1e-6))

	embed_dim = model.embed_dim
	out_dim = 128
	use_bn_in_head = False
	norm_last_layer = True
	clip_grad = 3.0
	freeze_last_layer = 0

	model = utils_transformer.MultiCropWrapper(model, vision_transformer.DINOHead(
		embed_dim,
		out_dim,
		use_bn=use_bn_in_head,
		norm_last_layer=norm_last_layer,
	))

	model.load_state_dict(torch.load(model_weights_filename), strict=False)

	model = model.backbone

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
					elif (ALGORITHM=='DINO'):
						feats = model(instances)

					feats = feats.view(-1, input_dim)
					feats_np = feats.cpu().data.numpy()
					
					features = np.append(features,feats_np)
					
					del instances
				#del instances
			features_np = np.reshape(features,(n_elems,input_dim))

			print(features_np.shape, len(csv_instances))
			#torch.cuda.empty_cache()
			del features, feats

			with open(filename_features, 'wb') as f:
				np.save(f, features_np)
		
		except FileNotFoundError as e:
			pass
			
torch.cuda.empty_cache()