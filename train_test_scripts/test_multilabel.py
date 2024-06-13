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
from sklearn import metrics 
from enum_multi import ALG, PHASE, SELF_SUPERVISION, TISSUE_TYPE

from model import CLAM_MB, CLAM_SB, Encoder, TransMIL, DSMIL, Additive_MIL, Attention_MIL
from utils import get_features_and_split, get_generator_instances, get_splits, generate_list_instances, save_prediction, save_loss_function, save_hyperparameters
from dataloader import Dataset_instance, Dataset_bag_multilabel, Balanced_Multimodal
from data_augmentation import generate_transformer
from metrics_multilabel import accuracy_micro, f1_scores, precisions, recalls

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
parser.add_argument('-d', '--dataset', help='dataset (AOEC/radboudumc)',type=str, default='AOEC')
parser.add_argument('-a', '--algorithm', help='ABMIL, ADMIL, CLAM, DSMIL, transMIL',type=str, default='CLAM')
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
EMBEDDING_bool = True
LABELS = args.LABELS
PREPROCESSED_DATA = args.preprocessed
DATASET = args.dataset
ALGORITHM = args.algorithm
WEIGHTS = args.weights
WEIGHTS_str = WEIGHTS
TISSUE = args.TISSUE
TISSUE_str = TISSUE

if (PREPROCESSED_DATA=='True'):
	PREPROCESSED_DATA = True
else:
	PREPROCESSED_DATA = False

hidden_space_len = args.hidden_space

seed = N_EXP
torch.manual_seed(seed)
if torch.cuda.is_available():
	torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

INPUT_folder = args.input_folder
OUTPUT_folder = args.output_folder
SELF_folder = args.self_supervised

print("PARAMETERS")
print("N_EPOCHS: " + str(EPOCHS_str))
print("CNN used: " + str(CNN_TO_USE))
print("BATCH_SIZE: " + str(BATCH_SIZE_str))
print("MAGNIFICATION: " + str(MAGNIFICATION))

####PATH where find patches
instance_dir = args.DATA_FOLDER

print("CREATE DIRECTORY WHERE MODELS WILL BE STORED")

models_path = OUTPUT_folder
os.makedirs(models_path, exist_ok=True)
models_path = models_path + TISSUE+'/'
os.makedirs(models_path, exist_ok=True)
models_path = models_path + '/multilabel/'
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
if (TISSUE=='Colon'):
	N_CLASSES = 5
elif (TISSUE=='Lung'):
	N_CLASSES = 4
	
print("CSV LOADING ")
csv_folder = args.CSV_FOLDER
csv_filename_testing = csv_folder+'ground_truth.csv'

test_dataset = pd.read_csv(csv_filename_testing, sep=',', header=None).values


#MODEL DEFINITION
#CNN BACKBONE
pre_trained_network = torch.hub.load('pytorch/vision:v0.10.0', CNN_TO_USE, pretrained=True)
if (('resnet' in CNN_TO_USE) or ('resnext' in CNN_TO_USE)):
	fc_input_features = pre_trained_network.fc.in_features
elif (('densenet' in CNN_TO_USE)):
	fc_input_features = pre_trained_network.classifier.in_features
elif ('mobilenet' in CNN_TO_USE):
	fc_input_features = pre_trained_network.classifier[1].in_features

input_feat = fc_input_features

		
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size_bag = 1

params_valid_bag = {'batch_size': 1,
		  'shuffle': False}


testing_set_bag = Dataset_bag_multilabel(test_dataset[:,0], test_dataset[:,1:])
testing_generator_bag = data.DataLoader(testing_set_bag, **params_valid_bag)

print("initialize CNN")
ALGORITHM = ALG[ALGORITHM]
WEIGHTS = SELF_SUPERVISION[WEIGHTS]
TISSUE = TISSUE_TYPE[TISSUE]

if (ALGORITHM is ALG.CLAM):
	model = CLAM_MB(n_classes = N_CLASSES, device = device)

elif (ALGORITHM is ALG.CLAM_MB):
	model = CLAM_MB(n_classes = N_CLASSES, device = device)

elif (ALGORITHM is ALG.CLAM_SB):
	model = CLAM_SB(n_classes = N_CLASSES, device = device)

elif (ALGORITHM is ALG.DSMIL):
	model = DSMIL(fc_input_features , hidden_space_len, N_CLASSES, device = device)

elif (ALGORITHM is ALG.transMIL):
	model = TransMIL(fc_input_features , hidden_space_len, N_CLASSES)

elif (ALGORITHM is ALG.ABMIL):
	model = Attention_MIL(N_CLASSES = N_CLASSES, fc_input_features = fc_input_features, hidden_space_len=128)

elif (ALGORITHM is ALG.ADMIL):
	model = Additive_MIL(N_CLASSES = N_CLASSES, fc_input_features = fc_input_features, hidden_space_len=128)

model.load_state_dict(torch.load(model_weights_filename), strict=False)
model.to(device)
model.eval()


#lr = 1e-4
num_epochs = EPOCHS

print("initialize hyperparameters")
import torch.optim as optim

print("testing")
print("testing at WSI level")
y_pred = []
y_true = []

mode = 'eval'
phase = 'test'

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

model.eval()

iterations = len(test_dataset)
dataloader_iterator = iter(testing_generator_bag)

with torch.no_grad():
	for i in range(iterations):
		print('%d / %d ' % (i, iterations))
		try:
			ID, labels = next(dataloader_iterator)
		except StopIteration:
			dataloader_iterator = iter(generator)
			ID, labels = next(dataloader_iterator)
			#inputs: bags, labels: labels of the bags

		ID = ID[0]
		filename_wsi = ID

		label_wsi = labels[0].cpu().data.numpy().flatten()
		labels_local = labels.float().flatten().to(device, non_blocking=True)

		print("[" + str(i) + "/" + str(iterations) + "], " + "inputs_bag: " + str(filename_wsi))
		print("labels: " + str(label_wsi))

		features_np = get_features_and_split(instance_dir, filename_wsi, WEIGHTS_str)

		inputs_embedding = torch.as_tensor(features_np).float().to(device, non_blocking=True)

		if (ALGORITHM is ALG.CLAM or ALGORITHM is ALG.CLAM_MB or ALGORITHM is ALG.CLAM_SB):
			logits_img, _ = model(inputs_embedding, None, instance_eval=False)

		else:
			logits_img = model(inputs_embedding)

		
			#loss img

		sigmoid_output_img = F.sigmoid(logits_img)
		outputs_wsi_np_img = sigmoid_output_img.cpu().data.numpy()
		
		print()
		print("pred_img: " + str(outputs_wsi_np_img))
		print()

		output_norm = np.where(outputs_wsi_np_img > 0.5, 1, 0)

		if (TISSUE is TISSUE_TYPE.Colon):
			filenames_wsis = np.append(filenames_wsis, filename_wsi)
			pred_cancers = np.append(pred_cancers, outputs_wsi_np_img[0])
			pred_hgd = np.append(pred_hgd, outputs_wsi_np_img[1])
			pred_lgd = np.append(pred_lgd, outputs_wsi_np_img[2])
			pred_hyper = np.append(pred_hyper, outputs_wsi_np_img[3])
			pred_normal = np.append(pred_normal, outputs_wsi_np_img[4])

		elif(TISSUE is TISSUE_TYPE.Lung):
			filenames_wsis = np.append(filenames_wsis, filename_wsi)
			pred_scc = np.append(pred_scc, outputs_wsi_np_img[0])
			pred_adeno = np.append(pred_adeno, outputs_wsi_np_img[1])
			pred_squamous = np.append(pred_squamous, outputs_wsi_np_img[2])
			pred_normal = np.append(pred_normal, outputs_wsi_np_img[3])

		y_pred = np.append(y_pred,output_norm)
		y_true = np.append(y_true,label_wsi)



if (TISSUE is TISSUE_TYPE.Colon):
	preds = np.stack((filenames_wsis, pred_cancers, pred_hgd, pred_lgd, pred_hyper, pred_normal), axis=1)

elif (TISSUE is TISSUE_TYPE.Lung):
	preds = np.stack((filenames_wsis, pred_scc, pred_adeno, pred_squamous, pred_normal), axis=1)

save_prediction(checkpoint_path, phase, None, preds, DATASET, TISSUE_str)

micro_accuracy = accuracy_micro(y_true, y_pred, None, None, None, DATASET)
f1_score_macro, f1_score_micro, f1_score_weighted = f1_scores(y_true, y_pred, i, N_CLASSES, checkpoint_path, phase, None, DATASET)
precision_score_macro, precision_score_micro = precisions(y_true, y_pred, i, N_CLASSES, checkpoint_path, phase, None, DATASET)
recall_score_macro, recall_score_micro = recalls(y_true, y_pred, i, N_CLASSES, checkpoint_path, phase, None, DATASET)

print("micro_accuracy " + str(micro_accuracy)) 
print("f1_score_macro " + str(f1_score_macro))
print("f1_score_micro " + str(f1_score_micro))
print("f1_score_weighted " + str(f1_score_weighted))
print("precision_score_macro " + str(precision_score_macro))
print("precision_score_micro " + str(precision_score_micro))
print("recall_score_macro " + str(recall_score_macro))
print("recall_score_micro " + str(recall_score_micro))

