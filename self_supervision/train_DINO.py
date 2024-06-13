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
from data_augmentation import DataAugmentationDINO
from dataloader import Dataset_instance, Dataset_instances_DINO
import torch.optim as optim
from loss_functions import DINOLoss
import vision_transformer
import utils_transformer 
from functools import partial
from torchvision import datasets, transforms
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
parser.add_argument('-l', '--lr', help='learning rate',type=float, default=5e-4)
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
min_lr = 1e-6

TISSUE = args.TISSUE
fp16_scaler = True

if (EMBEDDING_bool=='True'):
	EMBEDDING_bool = True
else:
	EMBEDDING_bool = False


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
models_path = models_path + 'DINO/'
os.makedirs(models_path, exist_ok=True)
model_weights_filename = models_path + 'DINO.pt'

#CSV LOADING
instance_dir = args.INPUT_IMAGES

print("CSV LOADING ")
k = 10

csv_filename_k_folds = args.WSI_LIST

#read data
data_split = pd.read_csv(csv_filename_k_folds, sep=',', header=None).values#[:10]

train_dataset, valid_dataset = utils.get_splits(data_split, N_EXP, TISSUE)

all_data = np.append(train_dataset, valid_dataset, axis=0)

print("training #: " + str(len(train_dataset)))
print("valid #: " + str(len(valid_dataset)))
print("all_data #: " + str(len(all_data)))

PATH_PATCHES = args.PATH_PATCHES
PATH_VALIDATION = args.PATH_VALIDATION

#all_data = valid_dataset
train_instances = utils.get_instances_paths_from_bags(instance_dir, train_dataset, PATH_PATCHES)
valid_instances = utils.get_instances_paths_from_bags(instance_dir, valid_dataset, PATH_VALIDATION)

#### load moco
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_dim = 512
hidden_dim = 512
output_dim = 128

patch_size = 16

student = vision_transformer.VisionTransformer(
		patch_size=patch_size, embed_dim=hidden_dim, depth=4, num_heads=8, mlp_ratio=4,
		qkv_bias=True, norm_layer=partial(torch.nn.LayerNorm, eps=1e-6))

teacher = vision_transformer.VisionTransformer(
		patch_size=patch_size, embed_dim=hidden_dim, depth=4, num_heads=8, mlp_ratio=4,
		qkv_bias=True, norm_layer=partial(torch.nn.LayerNorm, eps=1e-6))

embed_dim = student.embed_dim

out_dim = 128
use_bn_in_head = False
norm_last_layer = True
clip_grad = 3.0
freeze_last_layer = 0

student = utils_transformer.MultiCropWrapper(student, vision_transformer.DINOHead(
		embed_dim,
		out_dim,
		use_bn=use_bn_in_head,
		norm_last_layer=norm_last_layer,
	))

teacher = utils_transformer.MultiCropWrapper(
	teacher,
	vision_transformer.DINOHead(embed_dim, out_dim, use_bn_in_head),
)

teacher_without_ddp = teacher
teacher_without_ddp.load_state_dict(student.state_dict())
# there is no backpropagation through the teacher, so no need for gradients
for p in teacher.parameters():
	p.requires_grad = False


#### pre-processing pipeline
prob = 0.5
global_crops_scale = (0.4, 1.)
local_crops_number = 8
local_crops_scale = (0.05, 0.4)

transform = DataAugmentationDINO(
		global_crops_scale,
		local_crops_scale,
		local_crops_number,
	)


#get dataloaders
AUGMENT_PROB_THRESHOLD = 0.5
prob = AUGMENT_PROB_THRESHOLD

num_workers = 8
params_instance = {'batch_size': BATCH_SIZE,
		'shuffle': True,
		'pin_memory': True,
		'drop_last':True,
		'num_workers': num_workers}

training_set_bag = Dataset_instances_DINO(train_instances, transform)
training_generator = data.DataLoader(training_set_bag, **params_instance)

img_loss = 0.0
instance_loss = 0.0

best_loss = 100000.0

EARLY_STOP_NUM = 1000
early_stop_cont = 0
epoch = 0

batch_size_instance = int(BATCH_SIZE_str)

#### OPTIMIZER

momentum = 0.9

from optimizer_MoCo import LARS

params_groups = utils_transformer.get_params_groups(student)
"""
optimizer = LARS(student.parameters(), lr,
										weight_decay=wt_decay,
										momentum=momentum)
"""
#optimizer = LARS(params_groups) 
optimizer = torch.optim.AdamW(params_groups)

if (fp16_scaler):
	scaler = torch.cuda.amp.GradScaler()

ncrops = 8
warmup_teacher_temp = 0.04
teacher_temp = 0.04
warmup_teacher_temp_epochs = 0
student_temp = 0.1
center_momentum = 0.9
momentum_teacher = 0.996
iterations = int(len(train_instances) / BATCH_SIZE)+1

warmup_epochs = 0
lr_schedule = utils_transformer.cosine_scheduler(
		lr,  # linear scaling rule
		min_lr,
		EPOCHS, iterations,
		warmup_epochs=warmup_epochs,
	)

weight_decay_end = 0.4
#wt_decay = 1e-4
wt_decay = 0.04

wd_schedule = utils_transformer.cosine_scheduler(
		wt_decay,
		weight_decay_end,
		EPOCHS, iterations,
	)


momentum_schedule = utils_transformer.cosine_scheduler(momentum_teacher, 1,
											   EPOCHS, iterations)

criterion = DINOLoss(
		out_dim = out_dim,
		ncrops = ncrops + 2,  # total number of crops = 2 global crops + local_crops_number
		warmup_teacher_temp = warmup_teacher_temp,
		teacher_temp = teacher_temp,
		warmup_teacher_temp_epochs = warmup_teacher_temp_epochs,
		nepochs = EPOCHS,
	).to(device)

print("STARTING TRAINING")

teacher.to(device)
student.to(device)
teacher_without_ddp.to(device)

while (epoch<EPOCHS and early_stop_cont<EARLY_STOP_NUM):

	student.train()
	student.zero_grad(set_to_none=True)
	optimizer.zero_grad(set_to_none=True)	

	img_loss_1 = 0.0
	img_loss_2 = 0.0
	img_loss_3 = 0.0
	img_loss = 0.0

	dataloader_iterator = iter(training_generator)
	iterations = int(len(train_instances) / BATCH_SIZE)+1

	for i in range(iterations):
		
		student.zero_grad(set_to_none=True)
		optimizer.zero_grad(set_to_none=True)	

		it = iterations * epoch + i  # global training iteration

		for p, param_group in enumerate(optimizer.param_groups):
			param_group["lr"] = lr_schedule[it]
			if p == 0:  # only the first group is regularized
				param_group["weight_decay"] = wd_schedule[it]

		if (fp16_scaler):
			with torch.autocast(device_type='cuda', dtype=torch.float16):

				#print('[%d], %d / %d ' % (epoch, i, iterations))
				try:
					X = next(dataloader_iterator)
				except StopIteration:
					dataloader_iterator = iter(training_generator)
					X = next(dataloader_iterator)
				
				#X = X.to(device, non_blocking=True)
				X = [im.to(device, non_blocking=True) for im in X]

				teacher_output = teacher(X[:2])  # only the 2 global views pass through the teacher
				student_output = student(X)

				loss = criterion(student_output, teacher_output, epoch)

				param_norms = None
				scaler.scale(loss).backward()

				param_norms = utils_transformer.clip_gradients(student, clip_grad)
				utils_transformer.cancel_gradients_last_layer(epoch, student,
												freeze_last_layer)

				scaler.step(optimizer)
				scaler.update()

				with torch.no_grad():
					m = momentum_schedule[it]  # momentum parameter

					for param_q, param_k in zip(student.parameters(), teacher_without_ddp.parameters()):
						param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)


				student.zero_grad(set_to_none=True)
				optimizer.zero_grad(set_to_none=True)	

				img_loss = img_loss + ((1 / (i+1)) * (loss.item() - img_loss))
		
		else:
			#print('[%d], %d / %d ' % (epoch, i, iterations))
			try:
				X = next(dataloader_iterator)
			except StopIteration:
				dataloader_iterator = iter(training_generator)
				X = next(dataloader_iterator)
			
			#X = X.to(device, non_blocking=True)
			X = [im.to(device, non_blocking=True) for im in X]

			teacher_output = teacher(X[:2])  # only the 2 global views pass through the teacher
			student_output = student(X)

			loss = criterion(student_output, teacher_output, epoch)

			param_norms = None
			
			loss.backward()

			param_norms = utils_transformer.clip_gradients(student, clip_grad)
			utils_transformer.cancel_gradients_last_layer(epoch, student,
											freeze_last_layer)

			
			optimizer.step()

			with torch.no_grad():
				m = momentum_schedule[it]  # momentum parameter

				for param_q, param_k in zip(student.parameters(), teacher_without_ddp.parameters()):
					param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)


			student.zero_grad(set_to_none=True)
			optimizer.zero_grad(set_to_none=True)	

			img_loss = img_loss + ((1 / (i+1)) * (loss.item() - img_loss))

		if (i%100==0 and i!=0):
			print('[%d], %d / %d ' % (epoch, i, iterations))
			print("img_loss: " + str(img_loss))

			if (best_loss>img_loss):
				early_stop_cont = 0
				print ("=> Saving a new best model")
				print("previous loss : " + str(best_loss) + ", new loss function: " + str(img_loss))
				best_loss = img_loss

				try:
					torch.save(student.state_dict(), model_weights_filename,_use_new_zipfile_serialization=False)
				except:
					try:
						torch.save(student.state_dict(), model_weights_filename)
					except:
						torch.save(student, model_weights_filename)
				
			else:
				early_stop_cont = early_stop_cont+1

		#del X
		#torch.cuda.empty_cache()

	epoch = epoch + 1