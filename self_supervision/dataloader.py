import sys, getopt
import torch
from torch.utils import data
import numpy as np 
import pandas as pd
from PIL import Image
import warnings
warnings.filterwarnings("ignore")
#from pytorch_pretrained_bert.modeling import BertModel
import pyspng
from torchvision import transforms
from data_augmentation import DataAugmentationDINO

#sampler

class Dataset_instance_eval(data.Dataset):

	def __init__(self, list_IDs):
		self.list_IDs = list_IDs
		
		self.preprocess = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		])


	def __len__(self):
		return len(self.list_IDs)

	def __getitem__(self, index):
		# Select sample
		ID = self.list_IDs[index]#[0]
		# Load data and get label
		"""
		with open(ID, 'rb') as fin:
			X = pyspng.load(fin.read())
		"""
		try:
			X = Image.open(ID[0])
			#print("a")
		except:
			X = Image.open(ID)
			#print("b")
		X = np.asarray(X)
		
		#data transformation
		input_tensor = self.preprocess(X).type(torch.FloatTensor)

		#return input_tensor
		return input_tensor

class Dataset_instances_DINO(data.Dataset):

	def __init__(self, list_IDs, transform):
	
		self.list_IDs = list_IDs
		self.transform = transform
		
	def __len__(self):
	
		return len(self.list_IDs)
	
	def __getitem__(self, index):
	
		# Select sample
		ID = self.list_IDs[index]
		# Load data and get label
		try:
			X = Image.open(ID[0])
			#print("a")
		except:
			X = Image.open(ID)
			#print("b")
		images = self.transform(X)

		return images
		

class Dataset_instance(data.Dataset):

	def __init__(self, list_IDs, pipeline1, pipeline2):
		self.list_IDs = list_IDs
		self.pipeline_transform1 = pipeline1
		self.pipeline_transform2 = pipeline2
		
		self.preprocess = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		])


	def __len__(self):
		return len(self.list_IDs)

	def __getitem__(self, index):
		# Select sample
		ID = self.list_IDs[index]#[0]
		# Load data and get label
		"""
		with open(ID, 'rb') as fin:
			X = pyspng.load(fin.read())
		"""
		try:
			X = Image.open(ID[0])
			#print("a")
		except:
			X = Image.open(ID)
			#print("b")
		X = np.asarray(X)
		#img.close()

		X1 = self.pipeline_transform1(image=X)['image']
		X2 = self.pipeline_transform2(image=X)['image']

		#data transformation
		input_tensor = self.preprocess(X).type(torch.FloatTensor)
		input_tensor_aug1 = self.preprocess(X1).type(torch.FloatTensor)
		input_tensor_aug2 = self.preprocess(X2).type(torch.FloatTensor)

		#return input_tensor
		return input_tensor, input_tensor_aug1, input_tensor_aug2

#data loader at WSI-level
class Dataset_bag_multilabel(data.Dataset):

	def __init__(self, list_IDs, labels):

		self.labels = labels
		self.list_IDs = list_IDs
		
	def __len__(self):

		return len(self.list_IDs)

	def __getitem__(self, index):
		# Select sample
		ID = self.list_IDs[index]
		
		# Load data and get label
		
		y = self.labels[index]
		y = torch.as_tensor(y.tolist() , dtype=torch.float32)
		
		return ID, y

class Dataset_bag_multiclass(data.Dataset):

	def __init__(self, list_IDs, labels):

		self.labels = labels
		self.list_IDs = list_IDs
		
	def __len__(self):

		return len(self.list_IDs)

	def __getitem__(self, index):
		# Select sample
		ID = self.list_IDs[index]
		
		# Load data and get label
		
		y = self.labels[index]
		y = np.argmax(y)

		y = torch.as_tensor(y, dtype=torch.long)

		return ID, y

if __name__ == "__main__":
	pass