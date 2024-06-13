import sys, getopt
import torch
import numpy as np 
import pandas as pd
import os
import argparse
import warnings
import torch.nn as nn
warnings.filterwarnings("ignore")


class Encoder(torch.nn.Module):
	def __init__(self, CNN_TO_USE):
		"""
		In the constructor we instantiate two nn.Linear modules and assign them as
		member variables.
		"""
		super(Encoder, self).__init__()
		
		CNN_TO_USE = 'resnet34'

		pre_trained_network = torch.hub.load('pytorch/vision:v0.10.0', CNN_TO_USE, pretrained=True)
		if (('resnet' in CNN_TO_USE) or ('resnext' in CNN_TO_USE)):
			fc_input_features = pre_trained_network.fc.in_features
		elif (('densenet' in CNN_TO_USE)):
			fc_input_features = pre_trained_network.classifier.in_features
		elif ('mobilenet' in CNN_TO_USE):
			fc_input_features = pre_trained_network.classifier[1].in_features
	
		self.conv_layers = torch.nn.Sequential(*list(pre_trained_network.children())[:-1])
		self.fc_feat_in = fc_input_features
		
		if (torch.cuda.device_count()>1):
			self.conv_layers = torch.nn.DataParallel(self.conv_layers)

	def forward(self, x):

		conv_layers_out=self.conv_layers(x)

		features = conv_layers_out.view(-1, self.fc_feat_in)

		return features

class simCLR(nn.Module):

	def __init__(self, CNN_TO_USE, in_dim=512, out_dim=256, intermediate_dim=4096):
		
		super(simCLR, self).__init__()
		self.base_encoder = Encoder(CNN_TO_USE)

		self.activation = torch.nn.Tanh()
		self.activation = torch.nn.ReLU()

		self.in_dim = in_dim
		self.intermediate_dim = intermediate_dim
		self.out_dim = out_dim

		self.fc = torch.nn.Sequential(
			torch.nn.Linear(self.in_dim, self.intermediate_dim),
			torch.nn.BatchNorm1d(self.intermediate_dim),
			self.activation,
			torch.nn.Linear(self.intermediate_dim, self.out_dim),
			torch.nn.BatchNorm1d(self.out_dim)
		)

	def forward(self, x):

		x = self.base_encoder(x)
		x = self.fc(x)

		return x


class MoCo(nn.Module):
	def __init__(self, CNN_TO_USE, K=65536, m=0.999, T=0.07, in_dim=512, intermediate_dim = 128, out_dim = 128):
		'''
		dim : feature dimension (default: 128)
		K   : queue size; number of negative keys (default: 65536)
		m   : moco momentum of updating key encoder (default: 0.999)
		T   : softmax temperature (default: 0.07)
		'''
		super(MoCo, self).__init__()

		self.K = K
		self.m = m
		self.T = T

		self.in_dim = in_dim
		self.intermediate_dim = intermediate_dim
		self.out_dim = out_dim

		# create the encoders
		self.encoder_q = Encoder(CNN_TO_USE)
		self.encoder_k = Encoder(CNN_TO_USE)

		self.q_fc = torch.nn.Sequential(
			torch.nn.Linear(self.in_dim, self.intermediate_dim),
			torch.nn.ReLU(),
			torch.nn.Linear(self.intermediate_dim, self.out_dim),
		)

		self.k_fc = torch.nn.Sequential(
			torch.nn.Linear(self.in_dim, self.intermediate_dim),
			torch.nn.ReLU(),
			torch.nn.Linear(self.intermediate_dim, self.out_dim),
			#torch.nn.BatchNorm1d(self.out_dim)
		)

		for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
			param_k.data.copy_(param_q.data)  # initialize
			param_k.requires_grad = False     # not update by gradient

		for param_q, param_k in zip(self.q_fc.parameters(), self.k_fc.parameters()):
			param_k.data.copy_(param_q.data)  # initialize
			param_k.requires_grad = False     # not update by gradient

		# create the queue
		self.register_buffer('queue', torch.randn(out_dim, K))
		self.queue = nn.functional.normalize(self.queue, dim=0)

		self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

	@torch.no_grad()
	def _momentum_update_key_encoder(self):
		for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
			param_k.data = self.m * param_k.data + (1 - self.m) * param_q.data

		for param_q, param_k in zip(self.q_fc.parameters(), self.k_fc.parameters()):
			param_k.data = self.m * param_k.data + (1 - self.m) * param_q.data

	@torch.no_grad()
	def _dequeue_and_enqueue(self, keys):
		# gather keys before updating queue
		#keys = self._concat_all_gather(keys)

		batch_size = keys.shape[0]

		ptr = int(self.queue_ptr)
		assert self.K % batch_size == 0

		# dequeue and enqueue : replace the keys at ptr
		self.queue[:, ptr:ptr + batch_size] = keys.T
		ptr = (ptr + batch_size) % self.K

		self.queue_ptr[0] = ptr

	@torch.no_grad()
	def _batch_shuffle_ddp(self, x):

		idx = torch.randperm(x.size(0))
		x = x[idx]

		return x, idx


	@torch.no_grad()
	def _batch_unshuffle_ddp(self, x, idx_unshuffle):
		k_temp = torch.zeros_like(x)

		for a, j in enumerate(idx_unshuffle):
			k_temp[j] = x[a]
		x = k_temp

		return x

	def forward(self, img_q, img_k):
		'''
		Input:
			img_q: a batch of query images
			img_k: a batch of key images
		Output:
			logits, targets
		'''

		# compute query features
		q = self.encoder_q(img_q)  # (N, C)
		q = self.q_fc(q)

		#q = nn.functional.normalize(q, dim=1)

		# compute key features
		with torch.no_grad():
			self._momentum_update_key_encoder()

			# shuffle for making use of BN
			#img_k, idx_unshuffle = self._batch_shuffle_ddp(img_k)

			k = self.encoder_k(img_k)  # (N, C)
			k = self.k_fc(k)

			#k = nn.functional.normalize(k, dim=1)

			# undo shuffle
			#k = self._batch_unshuffle_ddp(k, idx_unshuffle)

		# compute positive, negative logits
		l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)  # (N, 1)
		l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])  # (N, K)
		logits = torch.cat([l_pos, l_neg], dim=1)  # (N, K+1)

		# apply temperature
		logits /= self.T

		# labels : positive key indicators
		labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)

		# dequeue and enqueue
		self._dequeue_and_enqueue(k)

		return logits, labels

if __name__ == "__main__":
	pass