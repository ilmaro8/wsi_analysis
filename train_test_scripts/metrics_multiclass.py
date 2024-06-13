import sys, getopt
import numpy as np 
import pandas as pd
from sklearn import metrics 
import os
import warnings
warnings.filterwarnings("ignore")
import sklearn
#from pytorch_pretrained_bert.modeling import BertModel

def save_metric(filename,value):
	array = [value]
	File = {'val':array}
	df = pd.DataFrame(File,columns=['val'])
	np.savetxt(filename, df.values, fmt='%s',delimiter=',')

def kappa_score(y_true, y_pred, checkpoint_path, phase, epoch, DATASET):

	kappa_score = metrics.cohen_kappa_score(y_true,y_pred, weights='quadratic')

	if (checkpoint_path is not None):
		if (phase=='test'):
			storing_dir = checkpoint_path + '/' + phase + '/'
		else:
			storing_dir = checkpoint_path + '/' + phase + '/epoch_' + str(epoch) + '/metrics/'
		os.makedirs(storing_dir, exist_ok = True)

		if (DATASET is None):
			filename = storing_dir + 'kappa_score.csv'
			save_metric(filename,kappa_score)

		else:
			filename = storing_dir + DATASET + '_kappa_score.csv'
			save_metric(filename,kappa_score)

	return kappa_score

def accuracy_score(y_true, y_pred, checkpoint_path, phase, epoch, DATASET):

	accuracy_score = metrics.accuracy_score(y_true,y_pred)

	if (checkpoint_path is not None):
		if (phase=='test'):
			storing_dir = checkpoint_path + '/' + phase + '/'
		else:
			storing_dir = checkpoint_path + '/' + phase + '/epoch_' + str(epoch) + '/metrics/'
		os.makedirs(storing_dir, exist_ok = True)

		if (DATASET is None):
			filename = storing_dir + 'accuracy_score.csv'
			save_metric(filename,accuracy_score)

		else:
			filename = storing_dir + DATASET + '_accuracy_score.csv'
			save_metric(filename,accuracy_score)

	return accuracy_score

def f1_scores(y_true, y_pred, checkpoint_path, phase, epoch, DATASET):
	
	f1_score_macro = metrics.f1_score(y_true=y_true, y_pred=y_pred, average='macro')
	f1_score_micro = metrics.f1_score(y_true=y_true, y_pred=y_pred, average='micro')
	f1_score_weighted = metrics.f1_score(y_true=y_true, y_pred=y_pred, average='weighted')

	if (checkpoint_path is not None):
		if (phase=='test'):
			storing_dir = checkpoint_path + '/' + phase + '/'
		else:
			storing_dir = checkpoint_path + '/' + phase + '/epoch_' + str(epoch) + '/metrics/'
		os.makedirs(storing_dir, exist_ok = True)
		
		if (DATASET is None):
			filename = storing_dir + 'f1_macro.csv'
			save_metric(filename,f1_score_macro)

			filename = storing_dir + 'f1_micro.csv'
			save_metric(filename,f1_score_micro)

			filename = storing_dir + 'f1_weighted.csv'
			save_metric(filename,f1_score_weighted)

		else:
			filename = storing_dir + DATASET + '_f1_macro_'+DATASET+'.csv'
			save_metric(filename,f1_score_macro)

			filename = storing_dir + DATASET + '_f1_micro_'+DATASET+'.csv'
			save_metric(filename,f1_score_micro)

			filename = storing_dir + DATASET + '_f1_weighted_'+DATASET+'.csv'
			save_metric(filename,f1_score_weighted)


	return f1_score_macro, f1_score_micro, f1_score_weighted

def recalls(y_true, y_pred, checkpoint_path, phase, epoch, DATASET):
	
	recall_score_macro = metrics.recall_score(y_true=y_true, y_pred=y_pred, average='macro')
	recall_score_micro = metrics.recall_score(y_true=y_true, y_pred=y_pred, average='micro')

	if (checkpoint_path is not None):
		if (phase=='test'):
			storing_dir = checkpoint_path + '/' + phase + '/'
		else:
			storing_dir = checkpoint_path + '/' + phase + '/epoch_' + str(epoch) + '/metrics/'
		os.makedirs(storing_dir, exist_ok = True)
		
		if (DATASET is None):

			filename = storing_dir + 'recall_score_macro.csv'
			save_metric(filename,recall_score_macro)

			filename = storing_dir + 'recall_score_micro.csv'
			save_metric(filename,recall_score_micro)

		else:
			filename = storing_dir + DATASET + '_recall_score_macro_'+DATASET+'.csv'
			save_metric(filename,recall_score_macro)

			filename = storing_dir + DATASET + '_recall_score_micro_'+DATASET+'.csv'
			save_metric(filename,recall_score_micro)


	return recall_score_macro, recall_score_micro

def precisions(y_true, y_pred, checkpoint_path, phase, epoch, DATASET):
	
	precision_score_macro = metrics.precision_score(y_true=y_true, y_pred=y_pred, average='macro')
	precision_score_micro = metrics.precision_score(y_true=y_true, y_pred=y_pred, average='micro')

	if (checkpoint_path is not None):
		if (phase=='test'):
			storing_dir = checkpoint_path + '/' + phase + '/'
		else:
			storing_dir = checkpoint_path + '/' + phase + '/epoch_' + str(epoch) + '/metrics/'
		os.makedirs(storing_dir, exist_ok = True)
		
		if (DATASET is None):

			filename = storing_dir + 'precision_score_macro.csv'
			save_metric(filename,precision_score_macro)

			filename = storing_dir + 'precision_score_micro.csv'
			save_metric(filename,precision_score_micro)

		else:
			filename = storing_dir + DATASET + '_precision_score_macro_'+DATASET+'.csv'
			save_metric(filename,precision_score_macro)

			filename = storing_dir + DATASET + '_precision_score_micro_'+DATASET+'.csv'
			save_metric(filename,precision_score_micro)

	return precision_score_macro, precision_score_micro

if __name__ == "__main__":
	pass