import sys, getopt
import numpy as np 
import pandas as pd
import os
import argparse
import warnings
warnings.filterwarnings("ignore")
import random
#from pytorch_pretrained_bert.modeling import BertModel
from torch.utils import data
from dataloader import Dataset_instance_eval 
from tqdm import tqdm

import random

def generate_list_instances(instance_dir, filename):
	fname = os.path.split(filename)[-1]
	
	#instance_csv = instance_dir+fname+'/'+fname+'_paths_densely_filter.csv'
	instance_csv = instance_dir+fname+'/'+fname+'_paths_densely.csv'
	try:
		data_split = pd.read_csv(instance_csv, sep=',', header=None).values#[:10]
	except:
		instance_csv = instance_dir+fname+'/'+fname+'_paths_densely.csv'

	return instance_csv 


def get_features_and_split(instance_dir, ID, MoCo_TO_USE):
	filename_features = instance_dir+ID+'/'+ID+'_features_'+MoCo_TO_USE+'.npy'

	with open(filename_features, 'rb') as f:
		features = np.load(f)
		f.close()

	n_indices = len(features)
	indices = np.random.choice(n_indices,n_indices,replace=False)
	
	#shuffled
	features = features[indices]
		
	return features

def get_generator_instances(csv_instances, batch_size_instance):

	#csv_instances = pd.read_csv(input_bag_wsi, sep=',', header=None).values
		#number of instances
	n_elems = len(csv_instances)
	
		#params generator instances
	

	num_workers = int(n_elems/batch_size_instance) + 1

	if (n_elems > batch_size_instance):
		pin_memory = True
	else:
		pin_memory = False

	params_instance = {'batch_size': batch_size_instance,
			'num_workers': num_workers,
			'pin_memory': pin_memory,
			'shuffle': True}

	instances = Dataset_instance_eval(csv_instances)
	generator = data.DataLoader(instances, **params_instance)

	return generator

def get_instances_paths_from_bags(instance_dir, dataset, PATH):

	print("loading data")
	try:
		fnames_patches = pd.read_csv(PATH, sep=',', header=None).values#[:10]

		print("loaded from file")
	except:

		print("generating on the fly")

		fnames_patches = []

		new_patches = 0	

		print('selecting all patches')

		for i in tqdm(range(len(dataset))):

			wsi = dataset[i]

			fname = wsi[0]

			csv_fname = generate_list_instances(instance_dir, fname)
			csv_instances = pd.read_csv(csv_fname, sep=',', header=None).values
			l_csv = len(csv_instances)
			
			#new_patches = new_patches + len(csv_instances)
			new_patches = new_patches + l_csv
			fnames_patches = np.append(fnames_patches, csv_instances)

		print(fnames_patches.shape)

		File = {'filenames':fnames_patches}
		df = pd.DataFrame(File,columns=['filenames'])
		np.savetxt(PATH, df.values, fmt='%s',delimiter=',')
	
	#print(fnames_patches.shape)
	fnames_patches = fnames_patches.tolist()

	return fnames_patches

def get_splits(data, n, TISSUE):
	
	train_dataset = []
	valid_dataset = []
	
	for sample in data:
		if (TISSUE == 'Colon'):

			fname = sample[0]
			cancer = sample[1]
			hgd = sample[2]
			lgd = sample[3]
			hyper = sample[4]
			normal = sample[5]
			f = sample[6]
		
			row = [fname, cancer, hgd, lgd, hyper, normal]
		
		elif (TISSUE == 'Lung'):
			
			fname = sample[0]
			scc = sample[1]
			adeno = sample[2]
			squamous = sample[3]
			normal = sample[4]
			f = sample[5]
		
			row = [fname, scc, adeno, squamous, normal]

		elif (TISSUE == 'Celiac'):
			
			fname = sample[0]
			celiac = sample[1]
			f = sample[2]
			
			row = [fname, celiac]


		if (f==n):
			
			valid_dataset.append(row)
		
		else:
			
			train_dataset.append(row)
			
	train_dataset = np.array(train_dataset, dtype=object)
	valid_dataset = np.array(valid_dataset, dtype=object)
	
	return train_dataset, valid_dataset

def save_prediction(checkpoint_path, phase, epoch, arrays, DATASET):
	
	if (phase=='test'):
		storing_dir = checkpoint_path + '/' + phase + '/'
	else:
		storing_dir = checkpoint_path + '/' + phase + '/epoch_' + str(epoch) + '/metrics/'

	os.makedirs(storing_dir, exist_ok = True)
	if (phase=='test'):
		filename_val = storing_dir+DATASET+'_predictions.csv'
	else:
		filename_val = storing_dir+'predictions.csv'
	
	File = {'filenames':arrays[:,0], 'pred_cancers':arrays[:,1], 'pred_hgd':arrays[:,2],'pred_lgd':arrays[:,3], 'pred_hyper':arrays[:,4], 'pred_normal':arrays[:,5]}

	df = pd.DataFrame(File,columns=['filenames','pred_cancers','pred_hgd','pred_lgd','pred_hyper','pred_normal'])
	np.savetxt(filename_val, df.values, fmt='%s',delimiter=',')

def save_loss_function(checkpoint_path, phase, epoch, value):

	storing_dir = checkpoint_path + '/' + phase + '/epoch_' + str(epoch) + '/'
	os.makedirs(storing_dir, exist_ok = True)

	filename_val = storing_dir+'loss_function.csv'
	array_val = [value]
	File = {'val':array_val}
	df = pd.DataFrame(File,columns=['val'])
	np.savetxt(filename_val, df.values, fmt='%s',delimiter=',')

def save_hyperparameters(checkpoint_path, N_CLASSES, EMBEDDING_bool, lr):

	filename_hyperparameters = checkpoint_path+'hyperparameters.csv'
	array_n_classes = [str(N_CLASSES)]
	array_lr = [str(lr)]
	array_embedding = [EMBEDDING_bool]
	File = {'n_classes':array_n_classes, 'lr':array_lr, 'embedding':array_embedding}
	df = pd.DataFrame(File,columns=['n_classes','lr','embedding'])
	np.savetxt(filename_hyperparameters, df.values, fmt='%s',delimiter=',')

if __name__ == "__main__":
	pass