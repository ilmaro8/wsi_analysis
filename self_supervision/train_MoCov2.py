import numpy as np
import pandas as pd
import os
from PIL import Image
import albumentations as A
import torch
from torch.utils import data
import torch.utils.data
import argparse
import warnings
import sys
import utils
from model import MoCo
from data_augmentation import get_pipeline_augment
from dataloader import Dataset_instance
import torch.optim as optim
from loss_functions import ContrastiveLoss, NT_Xent, SimCLR_Loss

warnings.filterwarnings("ignore")

torch.backends.cudnn.benchmark = True

argv = sys.argv[1:]

print("CUDA current device " + str(torch.cuda.current_device()))
print("CUDA devices available " + str(torch.cuda.device_count()))

#parser parameters
parser = argparse.ArgumentParser(description='Configurations to train models.')
parser.add_argument('-n', '--N_EXP', help='number of experiment',type=int, default=0)
parser.add_argument('-c', '--CNN', help='cnn_to_use',type=str, default='resnet34')
parser.add_argument('-b', '--BATCH_SIZE', help='batch_size',type=int, default=256)
parser.add_argument('-e', '--EPOCHS', help='epochs to train',type=int, default=10)
parser.add_argument('-m', '--MAG', help='magnification to select',type=str, default='10')
parser.add_argument('-f', '--features', help='features_to_use: embedding (True) or features from CNN (False)',type=str, default='True')
parser.add_argument('-l', '--lr', help='learning rate',type=float, default=1e-4)
parser.add_argument('-k', '--keys', help='number of keys',type=int, default=65536)
parser.add_argument('-t', '--TISSUE', help='tissue to use (Colon, Celiac, Lung)',type=str, default='Colon')
parser.add_argument('-o', '--OUTPUT_PATH', help='path of the folder where to store the trained model',type=str, default='')
parser.add_argument('-i', '--INPUT_IMAGES', help='path of the folder where WSI patches are stored',type=str, default='')
parser.add_argument('-p', '--PATH_PATCHES', help='path of the csv including all the patches for training',type=str, default='')
parser.add_argument('-v', '--PATH_VALIDATION', help='path of the csv including all the patches for validation',type=str, default='')
parser.add_argument('-w', '--WSI_LIST', help='path of the csv including the WSI ids and classes',type=str, default='')

args = parser.parse_args()

N_EXP = args.N_EXP
N_EXP_str = str(N_EXP)
CNN_TO_USE = args.CNN
BATCH_SIZE = args.BATCH_SIZE
BATCH_SIZE_str = str(BATCH_SIZE)
EPOCHS = args.EPOCHS
EPOCHS_str = EPOCHS
MAGNIFICATION = args.MAG
MAGNIFICATION_str = str(MAGNIFICATION)
EMBEDDING_bool = args.features
lr = args.lr
TISSUE = args.TISSUE


if (EMBEDDING_bool=='True'):
	EMBEDDING_bool = True
else:
	EMBEDDING_bool = False
	
num_keys = 4096
num_keys = 8192
num_keys = 16384
num_keys = 32768
num_keys = 65536
num_keys = args.keys

#print(EMBEDDING_bool)

seed = N_EXP
torch.manual_seed(seed)
if torch.cuda.is_available():
	torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

print("PARAMETERS")
print("N_EPOCHS: " + str(EPOCHS_str))
print("CNN used: " + str(CNN_TO_USE))
print("BATCH_SIZE: " + str(BATCH_SIZE_str))
print("MAGNIFICATION: " + str(MAGNIFICATION))


#DIRECTORIES CREATION
print("CREATING/CHECKING DIRECTORIES")

#models_path = '/PATH/TO/MOCO/MODEL'

models_path = args.OUTPUT_PATH
os.makedirs(models_path, exist_ok=True)
models_path = models_path + 'MoCo/'
os.makedirs(models_path, exist_ok=True)
models_path = models_path + CNN_TO_USE + '/'
os.makedirs(models_path, exist_ok=True)
model_weights_filename = models_path + 'MoCo.pt'

#CSV LOADING
instance_dir = args.INPUT_IMAGES

print("CSV LOADING ")
k = 10
N_CLASSES = 5
csv_filename_k_folds = args.WSI_LIST


#read data
data_split = pd.read_csv(csv_filename_k_folds, sep=',', header=None).values#[:10]

train_dataset, valid_dataset = utils.get_splits(data_split, N_EXP)

all_data = np.append(train_dataset, valid_dataset, axis=0)

print("training #: " + str(len(train_dataset)))
print("valid #: " + str(len(valid_dataset)))
print("all_data #: " + str(len(all_data)))

#all_data = valid_dataset
PATH_PATCHES = args.PATH_PATCHES
PATH_VALIDATION = args.PATH_VALIDATION

#all_data = valid_dataset
train_instances = utils.get_instances_paths_from_bags(instance_dir, train_dataset, PATH_PATCHES)
valid_instances = utils.get_instances_paths_from_bags(instance_dir, valid_dataset, PATH_VALIDATION)

#### load moco
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_dim = 512
hidden_dim = 128
output_dim = 128

momentum = 0.999
TEMPERATURE = 0.07

#encoder = simCLR(CNN_TO_USE, in_dim=input_dim, out_dim=output_dim, intermediate_dim=hidden_dim)
model = MoCo(CNN_TO_USE, K=num_keys, m=momentum, T=TEMPERATURE, in_dim=input_dim, intermediate_dim = hidden_dim, out_dim = output_dim)

model.eval()
model.to(device)
#model = DistributedDataParallel(model, device_ids=[device])
print(model)

#### pre-processing pipeline
pipeline_transform, pipeline_transform_soft = get_pipeline_augment()

#get dataloaders
AUGMENT_PROB_THRESHOLD = 0.5
prob = AUGMENT_PROB_THRESHOLD

num_workers = 8
params_instance = {'batch_size': BATCH_SIZE,
		'shuffle': True,
		'pin_memory': True,
		'drop_last':True,
		'num_workers': num_workers}

training_set_bag = Dataset_instance(train_instances, pipeline_transform, pipeline_transform_soft)
training_generator = data.DataLoader(training_set_bag, **params_instance)

img_loss = 0.0
instance_loss = 0.0

best_loss = 100000.0

EARLY_STOP_NUM = 10
early_stop_cont = 0
epoch = 0

batch_size_instance = int(BATCH_SIZE_str)

#### OPTIMIZER
lr = 1e-4
wt_decay = 1e-4
momentum = 0.9

from optimizer_MoCo import LARS
optimizer = LARS(model.parameters(), lr,
                                        weight_decay=wt_decay,
                                        momentum=momentum)
"""
optimizer = torch.optim.SGD(
        model.parameters(),
        lr,
        momentum=momentum,
        weight_decay=wt_decay,
    )
"""
scaler = torch.cuda.amp.GradScaler()


#criterion = ContrastiveLoss(TEMPERATURE)
#criterion = NT_Xent(batch_size=BATCH_SIZE, temperature=TEMPERATURE)
criterion = torch.nn.CrossEntropyLoss()

print("STARTING TRAINING")

while (epoch<EPOCHS and early_stop_cont<EARLY_STOP_NUM):

	model.train()
	model.zero_grad(set_to_none=True)
	optimizer.zero_grad(set_to_none=True)	

	img_loss_1 = 0.0
	img_loss_2 = 0.0
	img_loss_3 = 0.0
	img_loss = 0.0

	dataloader_iterator = iter(training_generator)
	iterations = int(len(train_instances) / BATCH_SIZE)+1

	for i in range(iterations):
		
		model.zero_grad(set_to_none=True)
		optimizer.zero_grad(set_to_none=True)	

		with torch.autocast(device_type='cuda', dtype=torch.float16):

			#print('[%d], %d / %d ' % (epoch, i, iterations))
			try:
				X, X1, X2 = next(dataloader_iterator)
			except StopIteration:
				dataloader_iterator = iter(training_generator)
				X, X1, X2 = next(dataloader_iterator)
			
			X = X.to(device, non_blocking=True)
			X1 = X1.to(device, non_blocking=True)
			X2 = X2.to(device, non_blocking=True)

			#q = encoder(X)
			q1, k1 = model(X1, X2)
			q2, k2 = model(X, X1)
			q, k = model(X, X2)
			#k1 = momentum_encoder(X1)
			#k2 = momentum_encoder(X2)

			loss1 = criterion(q1, k1)
			loss2 = criterion(q2, k2)
			loss3 = criterion(q, k)

			loss = loss1 + loss2 + loss3

			scaler.scale(loss).backward()
			scaler.step(optimizer)
			scaler.update()

			model.zero_grad(set_to_none=True)
			optimizer.zero_grad(set_to_none=True)	

			img_loss_1 = img_loss_1 + ((1 / (i+1)) * (loss1.item() - img_loss_1))
			img_loss_2 = img_loss_2 + ((1 / (i+1)) * (loss2.item() - img_loss_2))
			img_loss_3 = img_loss_3 + ((1 / (i+1)) * (loss3.item() - img_loss_3))
			img_loss = img_loss + ((1 / (i+1)) * (loss.item() - img_loss))
			

		if (i%100==0 and i!=0):
			print('[%d], %d / %d ' % (epoch, i, iterations))
			print("img_loss: " + str(img_loss))
			print("img_loss_1: " + str(img_loss_1))
			print("img_loss_2: " + str(img_loss_2))
			print("img_loss_3: " + str(img_loss_3))

			if (best_loss>img_loss):
				early_stop_cont = 0
				print ("=> Saving a new best model")
				print("previous loss : " + str(best_loss) + ", new loss function: " + str(img_loss))
				best_loss = img_loss

				try:
					torch.save(model.state_dict(), model_weights_filename,_use_new_zipfile_serialization=False)
				except:
					try:
						torch.save(model.state_dict(), model_weights_filename)
					except:
						torch.save(model, model_weights_filename)
				
			else:
				early_stop_cont = early_stop_cont+1