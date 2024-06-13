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
warnings.filterwarnings("ignore")

class NT_Xent(torch.nn.Module):
	def __init__(self, batch_size, temperature):
		super(NT_Xent, self).__init__()
		self.batch_size = batch_size
		self.temperature = temperature
		self.world_size = 1

		self.mask = self.mask_correlated_samples(batch_size, self.world_size)
		self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")
		self.similarity_f = torch.nn.CosineSimilarity(dim=2)

	def mask_correlated_samples(self, batch_size, world_size):
		N = 2 * batch_size * world_size
		mask = torch.ones((N, N), dtype=bool)
		mask = mask.fill_diagonal_(0)
		for i in range(batch_size * world_size):
			mask[i, batch_size * world_size + i] = 0
			mask[batch_size * world_size + i, i] = 0
		return mask

	def forward(self, z_i, z_j):
		"""
		We do not sample negative examples explicitly.
		Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
		"""
		N = 2 * self.batch_size * self.world_size

		z = torch.cat((z_i, z_j), dim=0)
		if self.world_size > 1:
			z = torch.cat(GatherLayer.apply(z), dim=0)

		sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

		sim_i_j = torch.diag(sim, self.batch_size * self.world_size)
		sim_j_i = torch.diag(sim, -self.batch_size * self.world_size)

		# We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
		positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
		negative_samples = sim[self.mask].reshape(N, -1)

		labels = torch.zeros(N).to(positive_samples.device).long()
		logits = torch.cat((positive_samples, negative_samples), dim=1)
		loss = self.criterion(logits, labels)
		loss /= N
		return loss

class NCELoss(torch.nn.Module):
	def __init__(self, embedding_size, num_samples):
		super(NCELoss, self).__init__()
		self.embedding_size = embedding_size
		self.num_samples = num_samples

		# define the weight matrix W and bias vector b as learnable parameters
		self.W = torch.nn.Parameter(torch.randn(self.embedding_size, self.embedding_size)).to(device)
		self.b = torch.nn.Parameter(torch.zeros(self.num_samples)).to(device)

		# sample noise from a uniform distribution over the vocabulary
		self.noise = torch.empty(self.num_samples, self.embedding_size).uniform_(-1, 1).to(device)

	def forward(self, input_tensor, target_tensor):
		"""
		:param input_tensor: a tensor of size (batch_size, embedding_size)
		:param target_tensor: a tensor of size (batch_size, embedding_size)
		:return: the NCE loss
		"""

		batch_size = input_tensor.size(0)

		# normalize the input and target tensors to have unit length
		input_norm = F.normalize(input_tensor, p=2, dim=1)
		target_norm = F.normalize(target_tensor, p=2, dim=1)

		# compute the dot product between the input and target tensors
		dot_product = torch.sum(input_norm * target_norm, dim=1)

		# compute the dot product between the input tensor and the noise tensor
		projected_input = input_norm @ self.W
		negative_scores = torch.sum(projected_input.view(batch_size, 1, self.embedding_size) *
									self.noise.view(1, self.num_samples, self.embedding_size), dim=2) + self.b

		# compute the log probability of the true samples and the noise samples
		true_log_prob = F.logsigmoid(dot_product)
		noise_log_prob = torch.sum(F.logsigmoid(-negative_scores), dim=1)

		# compute the average NCE loss across the batch
		loss = - (torch.sum(true_log_prob) + torch.sum(noise_log_prob)) / batch_size

		return loss

class SimCLR_Loss(nn.Module):
    def __init__(self, batch_size, temperature):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):

        N = 2 * self.batch_size

        z = torch.cat((z_i, z_j), dim=0)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)
        
        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)
        
        #SIMCLR
        labels = torch.from_numpy(np.array([0]*N)).reshape(-1).to(positive_samples.device).long() #.float()
        
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        
        return loss

class InfoNCELoss(torch.nn.Module):
	def __init__(self, temperature=0.07):
		super().__init__()
		self.temperature = temperature
	
	def forward(self, sample_embeddings, augmented_sample_embeddings):
		batch_size = sample_embeddings.shape[0]
		logits = torch.mm(sample_embeddings, augmented_sample_embeddings.T)
		labels = torch.arange(batch_size, dtype=torch.long).to(sample_embeddings.device)
		loss = F.cross_entropy(logits / self.temperature, labels)
		return loss

class ContrastiveLoss(torch.nn.Module):
	def __init__(self, temperature=0.07):
		super().__init__()
		self.T = temperature

	@torch.no_grad()
	def concat_all_gather(self, tensor):
		"""
		Performs all_gather operation on the provided tensors.
		*** Warning ***: torch.distributed.all_gather has no gradient.
		"""
		tensors_gather = [torch.ones_like(tensor)
			for _ in range(torch.distributed.get_world_size())]
		torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

		output = torch.cat(tensors_gather, dim=0)
		return output

	def forward(self, q, k):
		q = nn.functional.normalize(q, dim=1)
		k = nn.functional.normalize(k, dim=1)
		# gather all targets
		k = self.concat_all_gather(k)
		# Einstein sum is more intuitive
		logits = torch.einsum('nc,mc->nm', [q, k]) / self.T
		N = logits.shape[0]  # batch size per GPU
		labels = (torch.arange(N, dtype=torch.long) + N * torch.distributed.get_rank()).cuda()
		return nn.CrossEntropyLoss()(logits, labels) * (2 * self.T)


class DINOLoss(torch.nn.Module):
    def __init__(self, out_dim = 128, ncrops = 8, warmup_teacher_temp = 0.04, teacher_temp = 0.04,
                 warmup_teacher_temp_epochs = 0, nepochs = 10, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        batch_center = batch_center / (len(teacher_output))

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

if __name__ == "__main__":
	pass