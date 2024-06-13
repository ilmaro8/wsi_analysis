import sys, getopt
import torch
from torch.utils import data
import numpy as np 
import pandas as pd
import torch.nn.functional as F
import torch.utils.data
import os
import argparse
import warnings
import torch.nn as nn
from topk.svm import SmoothTop1SVM
warnings.filterwarnings("ignore")

from nystrom_attention import NystromAttention

#from pytorch_pretrained_bert.modeling import BertModel

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


########## attention based
class Attention_MIL(torch.nn.Module):
	def __init__(self, N_CLASSES, fc_input_features, hidden_space_len=128, TEMPERATURE = 0.07):
		"""
		In the constructor we instantiate two nn.Linear modules and assign them as
		member variables.
		"""
		super(Attention_MIL, self).__init__()

		#CNN_TO_USE = 'resnet34'

		self.N_CLASSES = N_CLASSES
		self.TEMPERATURE = TEMPERATURE

		self.input_feat = fc_input_features
		#embedding size
		self.E = hidden_space_len

		#attention network parameters
		self.L = self.E
		self.D = int(self.E / 2)
		#self.K = self.N_CLASSES 
		self.K = 1

		#######general components
		self.dropout = torch.nn.Dropout(p=0.1)
		self.tanh = torch.nn.Tanh()
		self.relu = torch.nn.ReLU()

		self.activation = self.relu

		#attention network
		self.attention = torch.nn.Sequential(
			torch.nn.Linear(self.L, self.D),
			torch.nn.Tanh(),
			torch.nn.Linear(self.D, self.K)
		)

		#from cnn embedding to intermediate embedding
		self.embedding = torch.nn.Linear(in_features=self.input_feat, out_features=self.E)
		
		self.embedding_after_attention = torch.nn.Linear(self.E * self.K, self.E)
		self.classifier = torch.nn.Linear(self.E, self.N_CLASSES)

	
	def forward(self, features):		
		A = None
		m_multiclass = torch.nn.Softmax()

		embedding_layer = self.embedding(features)
		embedding_layer = self.relu(embedding_layer)

		features_to_return = embedding_layer
		embedding_layer = self.dropout(features_to_return)

		A = self.attention(features_to_return)
		A = torch.transpose(A, 1, 0)
		A = F.softmax(A, dim=1)

		M = torch.mm(A, features_to_return)

		M = self.embedding_after_attention(M)
		M = self.activation(M)

		logits = self.classifier(M)
		#logits = logits.view(-1).view(-1)
		logits = logits.view(-1)
		
		return logits

########## additive MIL
class Additive_MIL(torch.nn.Module):
	def __init__(self, N_CLASSES, fc_input_features, hidden_space_len=128, TEMPERATURE = 0.07):
		"""
		In the constructor we instantiate two nn.Linear modules and assign them as
		member variables.
		"""
		super(Additive_MIL, self).__init__()

		#CNN_TO_USE = 'resnet34'

		self.N_CLASSES = N_CLASSES
		self.TEMPERATURE = TEMPERATURE

		self.input_feat = fc_input_features
		#embedding size
		self.E = hidden_space_len

		#attention network parameters
		self.L = self.E
		self.D = int(self.E / 2)
		#self.K = self.N_CLASSES 
		self.K = self.N_CLASSES

		#######general components
		self.dropout = torch.nn.Dropout(p=0.1)
		self.tanh = torch.nn.Tanh()
		self.relu = torch.nn.ReLU()

		self.activation = self.relu

		#attention network
		self.attention = torch.nn.Sequential(
			torch.nn.Linear(self.L, self.D),
			torch.nn.Tanh(),
			torch.nn.Linear(self.D, self.K)
		)

		#from cnn embedding to intermediate embedding
		self.embedding = torch.nn.Linear(in_features=self.input_feat, out_features=self.E)
		
		self.embedding_after_attention = torch.nn.Linear(self.E * self.K, self.E)
		self.classifier = torch.nn.Linear(self.E, self.N_CLASSES)

	
	def forward(self, features):		
		A = None
		m_multiclass = torch.nn.Softmax()

		embedding_layer = self.embedding(features)
		embedding_layer = self.relu(embedding_layer)

		features_to_return = embedding_layer
		embedding_layer = self.dropout(features_to_return)

		A = self.attention(features_to_return)
		A = torch.transpose(A, 1, 0)
		A = F.softmax(A, dim=1)

		M = torch.mm(A, features_to_return)

		M = M.view(-1, self.E * self.K)

		M = self.embedding_after_attention(M)
		M = self.activation(M)

		logits = self.classifier(M)

		logits = logits.view(-1)

		return logits

################ CLAM based

def initialize_weights(module):
	for m in module.modules():
		if isinstance(m, torch.nn.Linear):
			torch.nn.init.xavier_normal_(m.weight)
			m.bias.data.zero_()
		
		elif isinstance(m, torch.nn.BatchNorm1d):
			torch.nn.init.constant_(m.weight, 1)
			torch.nn.init.constant_(m.bias, 0)

class Attn_Net(torch.nn.Module):

	def __init__(self, L = 128, D = 128, dropout = False, n_classes = 1):
		super(Attn_Net, self).__init__()
		self.module = [
			torch.nn.Linear(L, D),
			torch.nn.Tanh()]

		if dropout:
			self.module.append(torch.nn.Dropout(0.25))

		self.module.append(torch.nn.Linear(D, n_classes))
		
		self.module = torch.nn.Sequential(*self.module)
	
	def forward(self, x):

		return self.module(x) # N x n_classes

"""
Attention Network with Sigmoid Gating (3 fc layers)
args:
	L: input feature dimension
	D: hidden layer dimension
	dropout: whether to use dropout (p = 0.25)
	n_classes: number of classes 
"""
class Attn_Net_Gated(torch.nn.Module):
	def __init__(self, L = 128, D = 128, dropout = False, n_classes = 1):
		super(Attn_Net_Gated, self).__init__()
		self.attention_a = [
			torch.nn.Linear(L, D),
			torch.nn.Tanh()]
		
		self.attention_b = [torch.nn.Linear(L, D),
							torch.nn.Sigmoid()]
		if dropout:
			self.attention_a.append(torch.nn.Dropout(0.25))
			self.attention_b.append(torch.nn.Dropout(0.25))

		self.attention_a = torch.nn.Sequential(*self.attention_a)
		self.attention_b = torch.nn.Sequential(*self.attention_b)
		
		self.attention_c = torch.nn.Linear(D, n_classes)

	def forward(self, x):
		a = self.attention_a(x)
		b = self.attention_b(x)
		A = a.mul(b)
		A = self.attention_c(A)  # N x n_classes
		return A, x		

class CLAM_SB(torch.nn.Module):
	def __init__(self, gate = False, size_arg = "small", dropout = True, k_sample=8, n_classes=2,
		instance_loss_fn=torch.nn.CrossEntropyLoss(), subtyping=False, device = None):
		
		super(CLAM_SB, self).__init__()
		#self.size_dict = {"small": [1024, 512, 256], "big": [1024, 512, 384]}
		self.size_dict = {"small": [512, 128, 128], "big": [512, 128, 128]}
		self.device = device
		size = self.size_dict[size_arg]
		
		self.fc = torch.nn.Linear(size[0], size[1])
		self.relu = torch.nn.ReLU()

		fc = [torch.nn.Linear(size[0], size[1]), torch.nn.ReLU()]
		
		self.dropout = torch.nn.Dropout(p=0.2)

		self.attention_net = Attn_Net(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
		self.classifiers = torch.nn.Linear(size[1], n_classes)
		
		instance_classifiers = [torch.nn.Linear(size[1], 2) for i in range(n_classes)]
		
		self.instance_classifiers = torch.nn.ModuleList(instance_classifiers)
		self.k_sample = k_sample

		if (instance_loss_fn == 'svm'):
			self.instance_loss_fn = SmoothTop1SVM(n_classes = 2).cuda()
		else:
			self.instance_loss_fn = torch.nn.CrossEntropyLoss()

		self.n_classes = n_classes
		self.subtyping = subtyping

		initialize_weights(self)

	def relocate(self):
		device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.attention_net = self.attention_net.to(device)
		self.instance_classifiers = self.instance_classifiers.to(device)
	
	@staticmethod
	def create_positive_targets(length, device):
		return torch.full((length, ), 1, device=device).long()
	@staticmethod
	def create_negative_targets(length, device):
		return torch.full((length, ), 0, device=device).long()
	
	#instance-level evaluation for in-the-class attention branch
	def inst_eval(self, A, h, classifier): 

		if len(A.shape) == 1:
			A = A.view(1, -1)
		top_p_ids = torch.topk(A, self.k_sample)[1][-1]
		top_p = torch.index_select(h, dim=0, index=top_p_ids)
		top_n_ids = torch.topk(-A, self.k_sample, dim=1)[1][-1]
		top_n = torch.index_select(h, dim=0, index=top_n_ids)
		p_targets = self.create_positive_targets(self.k_sample, self.device)
		n_targets = self.create_negative_targets(self.k_sample, self.device)

		all_targets = torch.cat([p_targets, n_targets], dim=0)
		all_instances = torch.cat([top_p, top_n], dim=0)

		logits = classifier(all_instances)
		all_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
		instance_loss = self.instance_loss_fn(logits, all_targets)

		return instance_loss, all_preds, all_targets
	
	#instance-level evaluation for out-of-the-class attention branch
	def inst_eval_out(self, A, h, classifier):

		if len(A.shape) == 1:
			A = A.view(1, -1)
		top_p_ids = torch.topk(A, self.k_sample)[1][-1]
		top_p = torch.index_select(h, dim=0, index=top_p_ids)
		p_targets = self.create_negative_targets(self.k_sample, self.device)
		logits = classifier(top_p)
		p_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
		instance_loss = self.instance_loss_fn(logits, p_targets)
		return instance_loss, p_preds, p_targets

	def forward(self, features, label=None, instance_eval=True, return_features=False, attention_only=False):

		h = self.fc(features)
		#h = self.relu(h)
		h = self.dropout(h)

		A = self.attention_net(h)  # NxK        
		A = torch.transpose(A, 1, 0)  # KxN
		A = F.softmax(A, dim=1)  # softmax over N

		total_inst_loss = 0.0

		try:
			if instance_eval:
				total_inst_loss = 0.0
				all_preds = []
				all_targets = []

				#inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze() #binarize label
				#inst_labels = label

				for i in range(len(self.instance_classifiers)):
					instance_loss = 0.0
					classifier = self.instance_classifiers[i]

					if (label[i]==1):

						instance_loss, preds, targets = self.inst_eval(A, h, classifier)
						all_preds.extend(preds.cpu().numpy())
						all_targets.extend(targets.cpu().numpy())
					else:
						if self.subtyping:

							instance_loss, preds, targets = self.inst_eval_out(A, h, classifier)
							all_preds.extend(preds.cpu().numpy())
							all_targets.extend(targets.cpu().numpy())

					total_inst_loss += instance_loss

				if self.subtyping:
					total_inst_loss /= len(self.instance_classifiers)

		except:
			total_inst_loss = 0.0
				
		M = torch.mm(A, h) 
		logits = self.classifiers(M)

		logits = torch.squeeze(logits)

		return logits, total_inst_loss

class CLAM_MB(CLAM_SB):

	def __init__(self, gate = False, size_arg = "small", dropout = True, k_sample=8, n_classes=4,
		instance_loss_fn = torch.nn.CrossEntropyLoss(), subtyping=False, device = None):

		torch.nn.Module.__init__(self)
		self.device = device

		#self.size_dict = {"small": [1024, 512, 256], "big": [1024, 512, 384]}
		self.size_dict = {"small": [512, 128, 128], "big": [1024, 128, 128]}

		size = self.size_dict[size_arg]

		self.fc = torch.nn.Linear(size[0], size[1])
		self.relu = torch.nn.ReLU()
		self.dropout = torch.nn.Dropout(p=0.2)

		self.attention_net = Attn_Net(L = size[1], D = size[2], dropout = dropout, n_classes = n_classes)
		bag_classifiers = [torch.nn.Linear(size[1], 1) for i in range(n_classes)] #use an indepdent linear layer to predict each class
		self.classifiers = torch.nn.ModuleList(bag_classifiers)

		#instance_classifier 
		instance_classifiers = [torch.nn.Linear(size[1], 2) for i in range(n_classes)]

		self.instance_classifiers = torch.nn.ModuleList(instance_classifiers)
		self.k_sample = k_sample

		if (instance_loss_fn == 'svm'):
			self.instance_loss_fn = SmoothTop1SVM(n_classes = 2).cuda()
		else:
			self.instance_loss_fn = torch.nn.CrossEntropyLoss()

		self.n_classes = n_classes
		self.subtyping = subtyping
		initialize_weights(self)

	def forward(self, features, label, instance_eval=True, return_features=False, attention_only=False):

		h = self.fc(features)
		#h = self.relu(h)
		h = self.dropout(h)

		A = self.attention_net(h)  # NxK        
		A = torch.transpose(A, 1, 0)  # KxN
		A = F.softmax(A, dim=1)  # softmax over N

		total_inst_loss = 0.0

		try:
			if instance_eval:
				total_inst_loss = 0.0
				all_preds = []
				all_targets = []

				#inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze() #binarize label
				#inst_labels = label

				for i in range(len(self.instance_classifiers)):
					instance_loss = 0.0
					classifier = self.instance_classifiers[i]

					if (label[i]==1):

						instance_loss, preds, targets = self.inst_eval(A[i], h, classifier)
						all_preds.extend(preds.cpu().numpy())
						all_targets.extend(targets.cpu().numpy())
					else:
						if self.subtyping:

							instance_loss, preds, targets = self.inst_eval_out(A[i], h, classifier)
							all_preds.extend(preds.cpu().numpy())
							all_targets.extend(targets.cpu().numpy())

					total_inst_loss += instance_loss

				if self.subtyping:
					total_inst_loss /= len(self.instance_classifiers)

		except:
			total_inst_loss = 0.0

		#logits pgp
		M = torch.mm(A, h) 
		logits = torch.empty(1, self.n_classes).float().to(self.device)
		
		for c in range(self.n_classes):
			logits[0, c] = self.classifiers[c](M[c])
		
		logits = torch.squeeze(logits)

		return logits, total_inst_loss

########## DSMIL
class IClassifier(torch.nn.Module):
	def __init__(self, input_size, feature_size, output_class):
		super(IClassifier, self).__init__()

		self.embedding = torch.nn.Linear(input_size, feature_size)
		self.fc = torch.nn.Linear(feature_size, output_class)
		
	def forward(self, x):
		
		feats = self.embedding(x)
		c = self.fc(feats) # N x C
		return feats.view(feats.shape[0], -1), c

class BClassifier(torch.nn.Module):

	def __init__(self, input_size, output_class, device, dropout_v=0.0, nonlinear=True, passing_v=False): # K, L, N
		super(BClassifier, self).__init__()
		if nonlinear:
			self.q = torch.nn.Sequential(torch.nn.Linear(input_size, 128), torch.nn.ReLU(), torch.nn.Linear(128, 128), torch.nn.Tanh())
		else:
			self.q = torch.nn.Linear(input_size, 128)
		if passing_v:
			self.v = torch.nn.Sequential(
				torch.nn.Dropout(dropout_v),
				torch.nn.Linear(input_size, input_size),
				torch.nn.ReLU()
			)
		else:
			self.v = torch.nn.Identity()
		
		### 1D convolutional layer that can handle multiple class (including binary)
		self.fcc = torch.nn.Conv1d(output_class, output_class, kernel_size=input_size)  
		
		self.device = device
		self.output_class = output_class

	def forward(self, feats, c): # N x K, N x C
		
		V = self.v(feats) # N x V, unsorted
		Q = self.q(feats).view(feats.shape[0], -1) # N x Q, unsorted
		
		# handle multiple classes without for loop
		_, m_indices = torch.sort(c, 0, descending=True) # sort class scores along the instance dimension, m_indices in shape N x C
		m_feats = torch.index_select(feats, dim=0, index=m_indices[0, :]) # select critical instances, m_feats in shape C x K 
		q_max = self.q(m_feats) # compute queries of critical instances, q_max in shape C x Q
		A = torch.mm(Q, q_max.transpose(0, 1)) # compute inner product of Q to each entry of q_max, A in shape N x C, each column contains unnormalized attention scores
		A = F.softmax( A / torch.sqrt(torch.tensor(Q.shape[1], dtype=torch.float32, device=self.device)), 0) # normalize attention scores, A in shape N x C, 
		B = torch.mm(A.transpose(0, 1), V) # compute bag representation, B in shape C x V
		B = B.view(1, B.shape[0], B.shape[1]) # 1 x C x V
		C = self.fcc(B) # 1 x C x 1
		#print(C.size())		
		#C = C.view(-1, self.output_class)
		C = torch.squeeze(C)

		return C#, A, B 
	
class DSMIL(torch.nn.Module):
	def __init__(self, fc_input_features, hidden_space_len, N_CLASSES, device):
		super(DSMIL, self).__init__()
		#self.i_classifier = i_classifier
		#self.b_classifier = b_classifier
		self.device = device

		self.input_features = fc_input_features
		self.E = hidden_space_len
		self.K = N_CLASSES

		self.i_classifier = IClassifier(self.input_features, self.E, N_CLASSES)
		
		self.b_classifier = BClassifier(self.E, N_CLASSES, device = self.device)
		

	def forward(self, x):

		feats, classes = self.i_classifier(x)

		#prediction_bag, A, B = self.b_classifier(feats, classes)
		prediction = self.b_classifier(feats, classes)

		#return classes, prediction_bag, A, B	
		return prediction

######### transMIL
class TransLayer(torch.nn.Module):

	def __init__(self, norm_layer=torch.nn.LayerNorm, dim=128):
		super().__init__()
		self.norm = norm_layer(dim)
		self.attn = NystromAttention(
			dim = dim,
			dim_head = dim//8,
			heads = 8,
			num_landmarks = dim//2,    # number of landmarks
			pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
			residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
			dropout=0.1
		)

	def forward(self, x):
		x = x + self.attn(self.norm(x))

		return x


class PPEG(torch.nn.Module):
	def __init__(self, dim=128):
		super(PPEG, self).__init__()
		self.proj = torch.nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
		self.proj1 = torch.nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
		self.proj2 = torch.nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)

	def forward(self, x, H, W):
		B, _, C = x.shape
		cls_token, feat_token = x[:, 0], x[:, 1:]
		cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
		x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)
		x = x.flatten(2).transpose(1, 2)
		x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
		return x


class TransMIL(torch.nn.Module):
	def __init__(self, fc_input_features, hidden_space_len, N_CLASSES):
		super(TransMIL, self).__init__()
		
		self.fc_feat_in = fc_input_features
		self.E = 128
		
		self.n_classes = N_CLASSES
		#pgp
		self.pos_layer = PPEG(dim=self.E)
		self._fc1 = torch.nn.Sequential(torch.nn.Linear(self.fc_feat_in, self.E), torch.nn.ReLU())
		
		self.cls_token = torch.nn.Parameter(torch.randn(1, 1, self.E))
		
		self.layer1 = TransLayer(dim=self.E)
		self.layer2 = TransLayer(dim=self.E)
		self.norm = torch.nn.LayerNorm(self.E)
		self._fc2 = torch.nn.Linear(self.E, self.n_classes)

		

	def forward(self, features):
		
		features = torch.reshape(features, (1, features.shape[0], features.shape[1]))

		#pgp
		h = self._fc1(features) #[B, n, 512]
		
		#---->pad
		H = h.shape[1]
		_H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
		add_length = _H * _W - H

		h = torch.cat([h, h[:,:add_length,:]],dim = 1) #[B, N, 512]

		#---->cls_token
		B = h.shape[0]
		cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
		h = torch.cat((cls_tokens, h), dim=1)

		#---->Translayer x1
		h = self.layer1(h) #[B, N, 512]

		#---->PPEG
		h = self.pos_layer(h, _H, _W) #[B, N, 512]
		
		#---->Translayer x2
		h = self.layer2(h) #[B, N, 512]

		#---->cls_token
		h = self.norm(h)[:,0]

		#---->predict
		logits = self._fc2(h) #[B, n_classes]

		logits = torch.squeeze(logits)

		return logits

if __name__ == "__main__":
	pass