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
import math 
from enum_multi import ALG, PHASE, SELF_SUPERVISION, TISSUE_TYPE

warnings.filterwarnings("ignore")

from nystrom_attention import NystromAttention

#from pytorch_pretrained_bert.modeling import BertModel

class MultiHeadAttention(torch.nn.Module):
	def __init__(self, d_model, num_heads, dropout = 0.1):
		super(MultiHeadAttention, self).__init__()
		assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
		
		self.d_model = d_model
		self.num_heads = num_heads
		self.d_k = d_model // num_heads
		
		#print("self.d_model, " + str(self.d_model) + ", self.num_heads: " + str(self.num_heads) + ", self.d_k: " + str(self.d_k))
		
		self.W_q = torch.nn.Linear(d_model, d_model)
		self.W_k = torch.nn.Linear(d_model, d_model)
		self.W_v = torch.nn.Linear(d_model, d_model)
		self.W_o = torch.nn.Linear(d_model, d_model)
		
		#print("self.W_q: " + str(self.W_q))
		#print("self.W_k: " + str(self.W_k))
		#print("self.W_v: " + str(self.W_v))
		#print("self.W_o: " + str(self.W_o))

		self.dropout = torch.nn.Dropout(dropout)
	
	def scaled_dot_product_attention(self, Q, K, V, mask=None):
		attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
		#print(attn_scores)
		if mask is not None:
			attn_scores = attn_scores.masked_fill(mask == False, -1e4)
			
		#print('aaaa')
		#print(attn_scores)
		attn_probs = torch.softmax(attn_scores, dim=-1)
		output = torch.matmul(self.dropout(attn_probs), V)
		return output
		
	def split_heads(self, x):
		batch_size, seq_length, d_model = x.size()
		return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
		
	def combine_heads(self, x):
		batch_size, _, seq_length, d_k = x.size()
		return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
		
	def forward(self, Q, K, V, mask=None):
		
		#print("Q: " + str(Q.shape))
		#print("K: " + str(K.shape))
		#print("V: " + str(V.shape))
		
		Wq = self.W_q(Q)
		Wk = self.W_k(K)
		Wv = self.W_v(V)
		
		#print("Wq: " + str(Wq.shape))
		#print("Wk: " + str(Wk.shape))
		#print("Wv: " + str(Wv.shape))
		
		Q = self.split_heads(Wq)
		K = self.split_heads(Wk)
		V = self.split_heads(Wv)
		
		#print("Q: " + str(Q.shape))
		#print("K: " + str(K.shape))
		#print("V: " + str(V.shape))
		#print()
		
		attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
		output = self.W_o(self.combine_heads(attn_output))
		return output, attn_output

class PositionWiseFeedForward(torch.nn.Module):
	def __init__(self, d_model, d_ff, dropout = 0.1):
		super(PositionWiseFeedForward, self).__init__()
		self.fc1 = torch.nn.Linear(d_model, d_ff)
		self.fc2 = torch.nn.Linear(d_ff, d_model)
		self.relu = torch.nn.ReLU()
		self.dropout = torch.nn.Dropout(dropout)
		
	def forward(self, x):
		
		x1 = self.fc1(x)
		x1 = self.relu(x1)
		x1 = self.dropout(x1)
		x1 = self.fc2(x1)
		
		return x1

class EncoderLayer(torch.nn.Module):
	def __init__(self, d_model, num_heads, d_ff, dropout):
		super(EncoderLayer, self).__init__()
		self.self_attn = MultiHeadAttention(d_model, num_heads)
		self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
		self.att_layer_norm = torch.nn.LayerNorm(d_model)
		self.ffn_layer_norm = torch.nn.LayerNorm(d_model)
		self.dropout = torch.nn.Dropout(dropout)
		
	def forward(self, x, mask):
		attn_output, attn_probs = self.self_attn(x, x, x, mask)
		x = self.att_layer_norm(x + self.dropout(attn_output))
		ff_output = self.feed_forward(x)
		x = self.ffn_layer_norm(x + self.dropout(ff_output))
		return x, attn_probs


class TransformerEncoder(torch.nn.Module):
	def __init__(self, d_model = 128, n_layers = 4, 
			   n_heads = 4, d_ffn = 512, dropout = 0.1):
		"""
		Args:
			d_model:      dimension of embeddings
			n_layers:     number of encoder layers
			n_heads:      number of heads
			d_ffn:        dimension of feed-forward network
			dropout:      probability of dropout occurring
		"""
		super().__init__()
		
		# create n_layers encoders 
		self.layers = torch.nn.ModuleList([EncoderLayer(d_model, n_heads, d_ffn, dropout)
									 for layer in range(n_layers)])
		
		self.dropout = torch.nn.Dropout(dropout)
	
	def forward(self, src, src_mask):
		"""
		Args:
			src:          embedded sequences                (batch_size, seq_length, d_model)
			src_mask:     mask for the sequences            (batch_size, 1, 1, seq_length)
		
		Returns:
			src:          sequences after self-attention    (batch_size, seq_length, d_model)
		"""
		
		# pass the sequences through each encoder
		for layer in self.layers:
			src, attn_probs = layer(src, src_mask)
		
		self.attn_probs = attn_probs
		
		return src

class PositionalEncoding(torch.nn.Module):
	def __init__(self, d_model, max_seq_length):
		super(PositionalEncoding, self).__init__()
		
		pe = torch.zeros(max_seq_length, d_model)
		position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
		
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		
		self.register_buffer('pe', pe.unsqueeze(0))
		
	def forward(self, x):
		x1 = x + self.pe[:, :x.size(1)].requires_grad_(False) 
		
		return x1

class Transformer(torch.nn.Module):
	def __init__(self, n_classes = 5, feat_size = 128, num_heads = 4, num_layers = 4, d_ff = 512, max_seq_length = 8832, dropout = 0.1, ALG = 'CNN_trans'):
		super(Transformer, self).__init__()
		
		print("feat_size: " + str(feat_size))
		print("num_heads: " + str(num_heads))
		print("num_layers: " + str(num_layers))
		print("d_ff: " + str(d_ff))
		print("max_seq_length: " + str(max_seq_length))
		
		self.ALG = ALG
		self.feat_size = feat_size
		self.seq_size = max_seq_length
		self.d_ff = d_ff
		self.num_heads = num_heads
		self.num_layers = num_layers
		self.n_classes = n_classes

		self.embedding_encoder = torch.nn.Linear(self.d_ff, self.feat_size)

		self.cls_token = torch.nn.Parameter(torch.zeros(1, self.feat_size))
		self.pos_enc = PositionalEncoding(self.feat_size, self.seq_size)
		self.encoder_layers = TransformerEncoder(self.feat_size, self.num_layers, self.num_heads, self.d_ff, dropout = 0.1)
		
		self.dropout = torch.nn.Dropout(dropout)
		self.activation = torch.nn.Tanh()
		self.classifier = torch.nn.Linear(self.feat_size, self.n_classes)
		
	def generate_mask(self, src):
		
		src_mask = (src.sum(dim=-1) != 0).unsqueeze(1).unsqueeze(2)

		return src_mask

	def prepare_tokens(self, x):
		
		new_token = torch.cat([self.cls_token.unsqueeze(0), x], dim=1)
		return new_token

	def padding(self, x):

		feat = x.size(2)
		elems = x.size(1)
	
		padding_needed = max(0, self.seq_size - elems)
	
		# Pad along the S dimension with -1
		padded_tensor = torch.nn.functional.pad(x, (0, 0, 0, padding_needed), value=0)
	
		return padded_tensor 
					   
	def forward(self, src):

		#embedding if not feat_size
		src = self.embedding_encoder(src)

		#cls token
		src = self.prepare_tokens(src)
		#padding
		src = self.padding(src)
		#mask
		src_mask = self.generate_mask(src)
		#print(src_mask)
		src_embedded = src    				
		
		src_embedded = self.pos_enc(src_embedded)
		#src_embedded = self.dropout(src_embedded)
		
		encoded_embedding_src = src_embedded
				
		enc_output = self.encoder_layers(encoded_embedding_src, src_mask)

		cls_token = enc_output[:,0,:]
		
		logits = self.classifier(cls_token)
		
		return logits

if __name__ == "__main__":
	pass