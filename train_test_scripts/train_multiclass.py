import sys, getopt
import torch
from torch.utils import data
import numpy as np
import pandas as pd
import torch.nn.functional as F
import os
import argparse
import warnings
warnings.filterwarnings("ignore")
from enum_multi import ALG, PHASE, SELF_SUPERVISION, TISSUE_TYPE

from model_multiclass import CLAM_MB, CLAM_SB, Encoder, TransMIL, DSMIL, Additive_MIL, Attention_MIL, MoCo, simCLR

from utils import get_features_and_split, get_generator_instances, get_splits, generate_list_instances, save_prediction, save_loss_function, save_hyperparameters
from dataloader import Dataset_instance, Dataset_bag_multiclass, Balanced_Multimodal
from data_augmentation import generate_transformer
from metrics_multiclass import accuracy_score, kappa_score, f1_scores, precisions, recalls
 
argv = sys.argv[1:]

print("CUDA current device " + str(torch.cuda.current_device()))
print("CUDA devices available " + str(torch.cuda.device_count()))

if torch.cuda.is_available():
	device = torch.device("cuda")
	print("working on gpu")
else:
	device = torch.device("cpu")
	print("working on cpu")
print(torch.backends.cudnn.version())

#parser parameters
parser = argparse.ArgumentParser(description='Configurations to train models.')
parser.add_argument('-n', '--N_EXP', help='number of experiment',type=int, default=0)
parser.add_argument('-c', '--CNN', help='cnn_to_use',type=str, default='resnet34')
parser.add_argument('-e', '--EPOCHS', help='epochs to train',type=int, default=10)
parser.add_argument('-m', '--MAG', help='magnification to select',type=str, default='10')
parser.add_argument('-l', '--LABELS', help='LABELS (GT/SKET)',type=str, default='SKET')
parser.add_argument('-p', '--preprocessed', help='pre-processed data: True False',type=str, default='True')
parser.add_argument('-z', '--hidden_space', help='hidden_space_size',type=int, default=128)
parser.add_argument('-a', '--algorithm', help='ABMIL, ADMIL, CLAM, DSMIL, transMIL',type=str, default='CLAM')
parser.add_argument('-b', '--batch_size', help='batch size bag level',type=int, default=4)
parser.add_argument('-i', '--input_folder', help='path folder input csv (there will be a train.csv file including IDs and labels)',type=str, default='')
parser.add_argument('-o', '--output_folder', help='path folder where to store output model',type=str, default='')
parser.add_argument('-s', '--self_supervised', help='path folder with pretrained network',type=str, default='')
parser.add_argument('-w', '--weights', help='algorithm for pre-trained weights',type=str, default='simCLR')
parser.add_argument('-t', '--TISSUE', help='tissue to use (Colon, Celiac, Lung)',type=str, default='Colon')
parser.add_argument('-d', '--DATA_FOLDER', help='path of the folder where to patches are stored',type=str, default='')
parser.add_argument('-f', '--CSV_FOLDER', help='folder where csv including IDs and classes are stored',type=str, default='True')

args = parser.parse_args()

N_EXP = args.N_EXP
N_EXP_str = str(N_EXP)
CNN_TO_USE = args.CNN
BATCH_SIZE = 512
BATCH_SIZE_str = str(BATCH_SIZE)
EPOCHS = args.EPOCHS
EPOCHS_str = EPOCHS
MAGNIFICATION = args.MAG
MAGNIFICATION_str = str(MAGNIFICATION)
LABELS = args.LABELS
WEIGHTS = args.weights
WEIGHTS_str = WEIGHTS
TISSUE = args.TISSUE
TISSUE_str = TISSUE

PREPROCESSED_DATA = args.preprocessed
ALGORITHM = args.algorithm
BATCH_SIZE_bag = args.batch_size

EMBEDDING_bool = True

if (PREPROCESSED_DATA=='True'):
	PREPROCESSED_DATA = True
else:
	PREPROCESSED_DATA = False

hidden_space_len = args.hidden_space

#TODO modify before uploading
INPUT_folder = args.input_folder
OUTPUT_folder = args.output_folder
SELF_folder = args.self_supervised


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
print("ALGORITHM: " + str(ALGORITHM))

####PATH where find patches
instance_dir = args.DATA_FOLDER

model_weights_filename_pre_trained = args.self_supervised

print("CREATE DIRECTORY WHERE MODELS WILL BE STORED")

models_path = OUTPUT_folder
os.makedirs(models_path, exist_ok=True)
models_path = models_path + TISSUE+'/'
os.makedirs(models_path, exist_ok=True)
models_path = models_path + '/multiclass/'
os.makedirs(models_path, exist_ok=True)
models_path = models_path+ALGORITHM+'/'
os.makedirs(models_path, exist_ok=True)
models_path = models_path+LABELS+'/'
os.makedirs(models_path, exist_ok=True)
if (PREPROCESSED_DATA):
	models_path = models_path+'no_augmentation/'
else:
	models_path = models_path+'augmentation/'
os.makedirs(models_path, exist_ok=True)
models_path = models_path+'magnification_'+MAGNIFICATION+'x/'
os.makedirs(models_path, exist_ok=True)
models_path = models_path+CNN_TO_USE+'/'
os.makedirs(models_path, exist_ok=True)
models_path = models_path+'N_EXP_'+N_EXP_str+'/'
os.makedirs(models_path, exist_ok=True)
checkpoint_path = models_path+'checkpoints_MIL/'
os.makedirs(checkpoint_path, exist_ok=True)

#path model file
model_weights_filename = models_path+'model.pt'

#CSV LOADING
print("CSV LOADING ")
k = 10

if (TISSUE=='Colon'):
	N_CLASSES = 5
elif (TISSUE=='Lung'):
	N_CLASSES = 4

csv_folder = args.CSV_FOLDER

if (LABELS=='SKET'):

	csv_filename_k_folds = csv_folder + str(k)+ '_cross_validation_sket.csv'

elif (LABELS=='GT'):

	csv_filename_k_folds = csv_folder + str(k)+ '_cross_validation_gt.csv'


#read data
data_split = pd.read_csv(csv_filename_k_folds, sep=',', header=None).values#[:10]

train_dataset, valid_dataset = get_splits(data_split, N_EXP, TISSUE)

#train_dataset = train_dataset[:20]
#valid_dataset = valid_dataset[:20]

#MODEL DEFINITION
#CNN BACKBONE
pre_trained_network = torch.hub.load('pytorch/vision:v0.10.0', CNN_TO_USE, pretrained=True)
if (('resnet' in CNN_TO_USE) or ('resnext' in CNN_TO_USE)):
	fc_input_features = pre_trained_network.fc.in_features
elif (('densenet' in CNN_TO_USE)):
	fc_input_features = pre_trained_network.classifier.in_features
elif ('mobilenet' in CNN_TO_USE):
	fc_input_features = pre_trained_network.classifier[1].in_features

	
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


AUGMENT_PROB_THRESHOLD = 0.5
prob = AUGMENT_PROB_THRESHOLD

sampler = Balanced_Multimodal

batch_size_bag = BATCH_SIZE_bag

if (PREPROCESSED_DATA==True):
	params_train_bag = {'batch_size': batch_size_bag,
		'shuffle': True}
else:
	params_train_bag = {'batch_size': batch_size_bag,
		'sampler': sampler(train_dataset,alpha=0.25)}

params_valid_bag = {'batch_size': batch_size_bag,
		  'shuffle': True}


training_set_bag = Dataset_bag_multiclass(train_dataset[:,0], train_dataset[:,1:])
training_generator_bag = data.DataLoader(training_set_bag, **params_train_bag)

validation_set_bag = Dataset_bag_multiclass(valid_dataset[:,0], valid_dataset[:,1:])
validation_generator_bag = data.DataLoader(validation_set_bag, **params_valid_bag)

print("initialize CNN")

input_dim = 512
hidden_dim = 128
output_dim = 128

ALGORITHM = ALG[ALGORITHM]
WEIGHTS = SELF_SUPERVISION[WEIGHTS]
TISSUE = TISSUE_TYPE[TISSUE]

try:

	encoder = Encoder(CNN_TO_USE)

	if (WEIGHTS is SELF_SUPERVISION.simCLR):
		pre_trained_backbone = simCLR(CNN_TO_USE, in_dim=input_dim, out_dim=output_dim, intermediate_dim=hidden_dim)

	elif (WEIGHTS is SELF_SUPERVISION.MoCo):
		momentum = 0.999
		TEMPERATURE = 0.07
		pre_trained_backbone = MoCo(CNN_TO_USE, K=num_keys, m=momentum, T=TEMPERATURE, in_dim=input_dim, intermediate_dim = hidden_dim, out_dim = output_dim)

	pre_trained_backbone.load_state_dict(torch.load(model_weights_filename_pre_trained), strict=False)

	if (WEIGHTS is SELF_SUPERVISION.simCLR):
		for param_e, param_p in zip(encoder.conv_layers.parameters(), pre_trained_backbone.base_encoder.parameters()):
			param_e.data.copy_(param_p.data)  # initialize
			param_e.requires_grad = False     # not 

	elif (WEIGHTS is SELF_SUPERVISION.MoCo):
		for param_e, param_p in zip(encoder.conv_layers.parameters(), pre_trained_backbone.encoder_q.parameters()):
			param_e.data.copy_(param_p.data)  # initialize
			param_e.requires_grad = False     # not 

	encoder.to(device)
	encoder.eval()
	print("moco loaded")

except Exception as e:
	print(e)
	encoder = Encoder(CNN_TO_USE)
	encoder.to(device)
	encoder.eval()
	print("new cnn")

for name, param in encoder.conv_layers.named_parameters():
	#if '10' in name or '11' in name: 
	param.requires_grad = False

encoder.eval()

instance_loss_fn = 'svm' 
#instance_loss_fn = 'ce' 

if (ALGORITHM is ALG.CLAM):
	model = CLAM_MB(n_classes = N_CLASSES, instance_loss_fn = instance_loss_fn, device = device)

elif (ALGORITHM is ALG.CLAM_MB):
	model = CLAM_MB(n_classes = N_CLASSES, instance_loss_fn = instance_loss_fn, device = device)

elif (ALGORITHM is ALG.CLAM_SB):
	model = CLAM_SB(n_classes = N_CLASSES, instance_loss_fn = instance_loss_fn, device = device)

elif (ALGORITHM is ALG.DSMIL):
	model = DSMIL(fc_input_features , hidden_space_len, N_CLASSES, device = device)

elif (ALGORITHM is ALG.transMIL):
	model = TransMIL(fc_input_features , hidden_space_len, N_CLASSES)

elif (ALGORITHM is ALG.ABMIL):
	model = Attention_MIL(N_CLASSES = N_CLASSES, fc_input_features = fc_input_features, hidden_space_len=128)

elif (ALGORITHM is ALG.ADMIL):
	model = Additive_MIL(N_CLASSES = N_CLASSES, fc_input_features = fc_input_features, hidden_space_len=128)

model.to(device)
model.eval()


#lr = 1e-4
num_epochs = EPOCHS

print("initialize hyperparameters")
import torch.optim as optim

if (ALGORITHM is ALG.CLAM or ALGORITHM is ALG.CLAM_MB or ALGORITHM is ALG.CLAM_SB):
	lr = 2e-4
	wt_decay = 1e-5
else:
	lr = 1e-4
	wt_decay = 1e-4


no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight",'prelu']
emb_par = ["img_embeddings_encoder.weight", "embedding_fc.weight"]

param_optimizer_CNN = list(model.named_parameters())
print(len(param_optimizer_CNN))
no_decay = ["prelu", "bias", "LayerNorm.bias", "LayerNorm.weight"]
optimizer_grouped_parameters_CNN = [
	{"params": [p for n, p in param_optimizer_CNN if not any(nd in n for nd in no_decay) and not any(nd in n for nd in emb_par)], "weight_decay": wt_decay},
	{"params": [p for n, p in param_optimizer_CNN if any(nd in n for nd in no_decay) and not any(nd in n for nd in emb_par)], "weight_decay": 0.0,},
	{"params": [p for n, p in param_optimizer_CNN if any(nd in n for nd in emb_par)], "weight_decay": wt_decay, 'lr': lr},
]

optimizer_CNN = optim.Adam(model.parameters(),lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=wt_decay, amsgrad=True)
#optimizer_CNN = optim.Adam(optimizer_grouped_parameters_CNN,lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=wt_decay, amsgrad=True)
#optimizer_CNN = AdamW(optimizer_grouped_parameters_CNN,lr = lr,eps=1e-8)

scaler = torch.cuda.amp.GradScaler()

criterion = torch.nn.CrossEntropyLoss()

def forward_features(generator_instance):

	features = []
	with torch.no_grad():
		for instances in generator_instance:
			instances = instances.to(device, non_blocking=True)

			# forward + backward + optimize
			feats = encoder(instances)
			feats = feats.view(-1, fc_input_features)
			feats_np = feats.cpu().data.numpy()
			
			features = np.append(features,feats_np)
	
	return features

def compute_features(instance_dir, ID):
	pipeline_transform = generate_transformer()
	instances_filename_sample = generate_list_instances(instance_dir, ID)

	csv_instances = pd.read_csv(instances_filename_sample, sep=',', header=None).values
	n_elems = len(csv_instances)

	generator_instance = get_generator_instances(csv_instances, 'train', pipeline_transform, batch_size_instance)
	encoder.eval()

	features = forward_features(generator_instance)
			
	features_np = np.reshape(features,(n_elems,fc_input_features))		

	return features_np

LAMBDA_INSTANCE = 0.2
EPOCH_THRESH_CLAM = 0

def model_forward(epoch, phase = 'train'):

	if (phase is PHASE.train):
		phase_str = 'train'
	elif (phase is PHASE.valid):
		phase_str = 'valid'

	img_loss = 0.0
	instance_loss = 0.0

	if (TISSUE is TISSUE_TYPE.Colon):
		filenames_wsis = []
		pred_cancers = []
		pred_hgd = []
		pred_lgd = []
		pred_hyper = []
		pred_normal = []

	elif (TISSUE is TISSUE_TYPE.Lung):
		filenames_wsis = []
		pred_scc = []
		pred_adeno = []
		pred_squamous = []
		pred_normal = []

	y_pred = []
	y_true = []

	

	if (phase is PHASE.train):
		
		dataloader_iterator = iter(training_generator_bag)
		iterations = iterations = int(len(train_dataset) / batch_size_bag)+1
		model.train()
		
	elif (phase is PHASE.valid):
		
		dataloader_iterator = iter(validation_generator_bag)
		iterations = len(valid_dataset)
		model.eval()
		

	for i in range(iterations):
		
		acc_instance_loss = 0.0

		with torch.autocast(device_type='cuda', dtype=torch.float16):

			print('[%d], %d / %d ' % (epoch, i, iterations))
			try:
				IDs, labels = next(dataloader_iterator)
			except StopIteration:
				dataloader_iterator = iter(training_generator_bag)
				IDs, labels = next(dataloader_iterator)
			
			array_logits_img = []

			labels = labels.to(device, non_blocking=True)

			for x, filename_wsi in enumerate(IDs):
				
				labels_np = labels[x].cpu().data.numpy().flatten()

				prob_pre = np.random.rand(1)[0]
				
				if (phase is PHASE.train and PREPROCESSED_DATA==False and prob_pre>=AUGMENT_PROB_THRESHOLD):

					features_np = compute_features(instance_dir, filename_wsi)

				else:
					
					features_np = get_features_and_split(instance_dir, filename_wsi, WEIGHTS_str)
			

				if (phase is PHASE.train):
					model.train()
					model.zero_grad(set_to_none=True)
					optimizer_CNN.zero_grad(set_to_none=True)	

					inputs_embedding = torch.tensor(features_np, requires_grad=True).float().to(device, non_blocking=True)

					if (ALGORITHM is ALG.CLAM or ALGORITHM is ALG.CLAM_MB or ALGORITHM is ALG.CLAM_SB):
						logits_img, instance_eval_loss = model(inputs_embedding, labels[x], instance_eval=True)
						acc_instance_loss = acc_instance_loss + instance_eval_loss / batch_size_bag
						
					else: 

						logits_img = model(inputs_embedding)
				
					array_logits_img.append(logits_img)

				else:
					model.eval()
					inputs_embedding = torch.as_tensor(features_np).float().to(device, non_blocking=True)

					with torch.no_grad():
						
						if (ALGORITHM is ALG.CLAM or ALGORITHM is ALG.CLAM_MB or ALGORITHM is ALG.CLAM_SB):
							logits_img, _ = model(inputs_embedding, None, instance_eval=False)

						else:
							logits_img = model(inputs_embedding)

				
					array_logits_img.append(logits_img)

				softmax_output_img = F.softmax(logits_img)
				outputs_wsi_np_img = softmax_output_img.cpu().data.numpy()

				if (TISSUE is TISSUE_TYPE.Colon):
					filenames_wsis = np.append(filenames_wsis, filename_wsi)
					pred_cancers = np.append(pred_cancers, outputs_wsi_np_img[0])
					pred_hgd = np.append(pred_hgd, outputs_wsi_np_img[1])
					pred_lgd = np.append(pred_lgd, outputs_wsi_np_img[2])
					pred_hyper = np.append(pred_hyper, outputs_wsi_np_img[3])
					pred_normal = np.append(pred_normal, outputs_wsi_np_img[4])

				elif (TISSUE is TISSUE_TYPE.Lung):
					filenames_wsis = np.append(filenames_wsis, filename_wsi)
					pred_scc = np.append(pred_scc, outputs_wsi_np_img[0])
					pred_adeno = np.append(pred_adeno, outputs_wsi_np_img[1])
					pred_squamous = np.append(pred_squamous, outputs_wsi_np_img[2])
					pred_normal = np.append(pred_normal, outputs_wsi_np_img[3])

				output_norm = np.argmax(outputs_wsi_np_img)

				y_pred = np.append(y_pred,output_norm)
				y_true = np.append(y_true,labels_np)

			array_logits_img = torch.stack(array_logits_img, dim=0).to(device)
			loss_img = criterion(array_logits_img, labels)

			if (phase is PHASE.train):
				
				if ((ALGORITHM is ALG.CLAM or ALGORITHM is ALG.CLAM_MB or ALGORITHM is ALG.CLAM_SB) and epoch >= EPOCH_THRESH_CLAM):
					loss = loss_img + LAMBDA_INSTANCE * acc_instance_loss
				else:
					loss = loss_img

				scaler.scale(loss).backward()
				scaler.step(optimizer_CNN)
				scaler.update()
				optimizer_CNN.zero_grad(set_to_none=True)
				model.zero_grad(set_to_none=True)

				if (ALGORITHM is ALG.CLAM or ALGORITHM is ALG.CLAM_MB or ALGORITHM is ALG.CLAM_SB):
					try:
						instance_loss = instance_loss + ((1 / (i+1)) * (acc_instance_loss.item() - instance_loss))
					except:
						print(acc_instance_loss)

			else:
				loss = loss_img

			img_loss = img_loss + ((1 / (i+1)) * (loss.item() - img_loss))

			print("img_loss: " + str(img_loss))

			if ((ALGORITHM is ALG.CLAM or ALGORITHM is ALG.CLAM_MB or ALGORITHM is ALG.CLAM_SB) and phase is PHASE.train):

				print("instance_loss: " + str(instance_loss))

			

			if (i == (iterations - 1)):
				micro_accuracy_train = accuracy_score(y_true, y_pred, checkpoint_path, phase_str, epoch, None)
				kappa = kappa_score(y_true, y_pred, checkpoint_path, phase_str, epoch, None)
				f1_score_macro, f1_score_micro, f1_score_weighted = f1_scores(y_true, y_pred, checkpoint_path, phase_str, epoch, None)
				precision_score_macro, precision_score_micro = precisions(y_true, y_pred, checkpoint_path, phase_str, epoch, None)
				recall_score_macro, recall_score_micro = recalls(y_true, y_pred, checkpoint_path, phase_str, epoch, None)

			else:
				micro_accuracy_train = accuracy_score(y_true, y_pred, None, None, None, None)
				kappa = kappa_score(y_true, y_pred, None, None, None, None)
				f1_score_macro, f1_score_micro, f1_score_weighted = f1_scores(y_true, y_pred, None, None, None, None)
				precision_score_macro, precision_score_micro = precisions(y_true, y_pred, None, None, None, None)
				recall_score_macro, recall_score_micro = recalls(y_true, y_pred, None, None, None, None)

			if (i == (iterations - 1)):
				
				print("accuracy " + str(micro_accuracy_train)) 
				print("kappa " + str(kappa)) 
				print("f1_score_macro " + str(f1_score_macro))
				print("f1_score_micro " + str(f1_score_micro))
				print("f1_score_weighted " + str(f1_score_weighted))
				print("precision_score_macro " + str(precision_score_macro))
				print("precision_score_micro " + str(precision_score_micro))
				print("recall_score_macro " + str(recall_score_macro))
				print("recall_score_micro " + str(recall_score_micro))

				#save pred
				if (TISSUE is TISSUE_TYPE.Colon):
					preds = np.stack((filenames_wsis, pred_cancers, pred_hgd, pred_lgd, pred_hyper, pred_normal), axis=1)

				elif (TISSUE is TISSUE_TYPE.Lung):
					preds = np.stack((filenames_wsis, pred_scc, pred_adeno, pred_squamous, pred_normal), axis=1)

				save_prediction(checkpoint_path, phase_str, epoch, preds, None, TISSUE_str)

				#save loss
				save_loss_function(checkpoint_path, phase_str, epoch, img_loss)
			
			print()

	return img_loss


epoch = 0

best_loss = 100000.0

EARLY_STOP_NUM = 10
early_stop_cont = 0
epoch = 0

batch_size_instance = int(BATCH_SIZE_str)

while (epoch<num_epochs and early_stop_cont<EARLY_STOP_NUM):
		
	#train
	phase = PHASE['train']
	train_loss = model_forward(epoch, phase = phase)

	phase = PHASE['valid']
	valid_loss = model_forward(epoch, phase = phase)
	
	if (best_loss>valid_loss):
		early_stop_cont = 0
		print ("=> Saving a new best model")
		print("previous loss : " + str(best_loss) + ", new loss function: " + str(valid_loss))
		best_loss = valid_loss

		try:
			torch.save(model.state_dict(), model_weights_filename,_use_new_zipfile_serialization=False)
		except:
			try:
				torch.save(model.state_dict(), model_weights_filename)
			except:
				torch.save(model, model_weights_filename)
		
	else:
		early_stop_cont = early_stop_cont+1
	
	#save hyper
	save_hyperparameters(checkpoint_path, N_CLASSES, EMBEDDING_bool, lr)
	
	epoch = epoch + 1

torch.cuda.empty_cache()