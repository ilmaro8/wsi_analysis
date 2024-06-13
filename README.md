# wsi_analysis

A full pipeline to pre-process WSIs and to analyze them (the selected use case is classification). The classification is made at WSI-level, involving binary, multiclass and multilabel data. Multiple classification algorithms are adopted.

## Reference
If you find this repository useful in your research, please cite:

[1] XXX.

Paper link: XXX


## Requirements
Python==3.6.9, albumentations==0.1.8, numpy==1.17.3, opencv==4.2.0, pandas==0.25.2, pillow==6.1.0, torchvision==0.8.1, pytorch==1.7.0

## input data
- csv files: csv files include information about input data. When they refer to the WSIs, they include a column with the image IDs; the class labels, reported as one-hot encoding; the folder to which the sample belong (in case the k-fold cross-validation is adopted); a fake patient ID (an integer).
- WSIs: the multi-level pyramid file including the histopathology image and its metadata.
- Masks: the tissue mask generated using HistoQC.
- patches: small slices of the WSIs in .jpg or .png format

## automatic weak labels
The code used to automatically extract meaningful concepts from the reports, used as weak labels, is available at: https://github.com/ExaNLP/sket.

## pre-processing
The folder includes a script to extract patches from a WSI (based on [Multi_Scale_Tools](https://github.com/sara-nl/multi-scale-tools).
- patch_extractiong.py: given tissue masks (already generated with HistoQC), the WSI is split in patches. The resolution of the patches depends on the selected magnification level where patches are extracted and on the patch size (e.g. patches of size 224x224 from magnification 10x). The script includes the following parameters:
  * -c (--USE_CASE): the type of tissue (in this paper celiac disease, colon cancer, lung cancer)
  * -m (--MAGNIFICATION): the magnification level from which extract the patches
  * -t (--THREADING): the adoption of python's multithreading or multiprocess
  * -i (--INPUT_DATA): the path of the csv including the WSI paths (absolute paths)
  * -o (--OUTPUT_PATH): the path of the folder where the extracted patches will be stored. The folder will include many subfolders, one for each WSI
  * -l (--MASKS_PATH): the path where the tissue masks (images) are stored
  

## self-supervision
The folder includes script to pre-train a neural network with self-supervised algorithms. The algorithms are: MoCo and simCLR (to pre-train CNNs) and DINO (to pre-train a ViT). The folder also includes many modules containing methods helpful for the pre-training. Furthermore, the folder includes scripts to extract and store features from the input data, according to the selected algorithm, so that the features can be loaded instead of the single patches, dramatically accelerating the training.

- train_MoCo.py
  * -n (--N_EXP): the number of repetitions (in case multiple versions of the same model are trained).
  * -c (--CNN): CNN backbone to use (loading the model and the weights with PyTorch methods and using ImageNet pre-trained weights).
  * -b (--BATCH_SIZE): batch size to use.
  * -e (--EPOCHS): epochs to train the model.
  * -m (--MAG): magnification of input data.
  * -l (--lr): learning rate.
  * -t (--TISSUE): tissue type (in this paper Colon, Celiac disease, Lung).
  * -o (--OUTPUT_PATH): folder where to store the model weights.
  * -i (--INPUT_IMAGES): path of the folder where WSI patches are stored.
  * -p (--PATH_PATCHES): path of the csv including all the patches for training.
  * -v (--PATH_VALIDATION): path of the csv including all the patches for validation.
  * -w (--WSI_LIST): path of the csv including the WSI ids and classes.
  * -k ('--keys'): the size of the queue including examples.

- train_simclr.py
  * -n (--N_EXP): the number of repetitions (in case multiple versions of the same model are trained).
  * -c (--CNN): CNN backbone to use (loading the model and the weights with PyTorch methods and using ImageNet pre-trained weights).
  * -b (--BATCH_SIZE): batch size to use.
  * -e (--EPOCHS): epochs to train the model.
  * -m (--MAG): magnification of input data.
  * -l (--lr): learning rate.
  * -t (--TISSUE): tissue type (in this paper Colon, Celiac disease, Lung).
  * -o (--OUTPUT_PATH): folder where to store the model weights.
  * -i (--INPUT_IMAGES): path of the folder where WSI patches are stored.
  * -p (--PATH_PATCHES): path of the csv including all the patches for training.
  * -v (--PATH_VALIDATION): path of the csv including all the patches for validation.
  * -w (--WSI_LIST): path of the csv including the WSI ids and classes.

 -train_DINO.py
  * -n (--N_EXP): the number of repetitions (in case multiple versions of the same model are trained).
  * -b (--BATCH_SIZE): batch size to use.
  * -e (--EPOCHS): epochs to train the model.
  * -m (--MAG): magnification of input data.
  * -l (--lr): learning rate.
  * -t (--TISSUE): tissue type (in this paper Colon, Celiac disease, Lung).
  * -o (--OUTPUT_PATH): folder where to store the model weights.
  * -i (--INPUT_IMAGES): path of the folder where WSI patches are stored.
  * -p (--PATH_PATCHES): path of the csv including all the patches for training.
  * -v (--PATH_VALIDATION): path of the csv including all the patches for validation.
  * -w (--WSI_LIST): path of the csv including the WSI ids and classes.

-feature_extractor scripts: this script aims to pre-process and store the features of the patches corresponding to WSIs, producing a .npy file for every WSI, which is (x,f) in size, where x is the amount of patches and f the feature size of the backbone output. The goal of this script is to pre-evaluate the features corresponding to patches to accelerate the training of MIL algorithms. Most of the time, MIL algorithms show a frozen backbone. Pre-evaluating the output of the backbone for the patches avoid to load and unload millions of patches. On the other hand, this solution prevents the adoption of data augmentation techniques.

-feature_extractor.py
 * -n (--N_EXP): number of the experiment/repetition to load the weights, in case multiple models were pre-trained.
 * -c (--CNN): CNN backbone to use (must match with the weights loaded).
 * -m (--MAG): magnification of input data.
 * -a (--ALGORITHM): algorithm used to pre-train the backbone (MoCo, simCLR).
 * -t (--TISSUE): tissue to use (Colon, Celiac, Lung)
 * -p (--PATH_MODEL): path of the file including model weights
 * -i (--FOLDER_DATA): folder where patches are stored
 * -l (--LIST_IMAGES): path of the csv file including the WSI IDs (used to create the input file with patch paths, in case it s missing.
 * -k (--keys): the size of the queue including examples.

-feature_extractor.py
 * -n (--N_EXP): number of the experiment/repetition to load the weights, in case multiple models were pre-trained.
 * -c (--CNN): CNN backbone to use (must match with the weights loaded).
 * -m (--MAG): magnification of input data.
 * -a (--ALGORITHM): algorithm used to pre-train the backbone (MoCo, simCLR).
 * -t (--TISSUE): tissue to use (Colon, Celiac, Lung)
 * -p (--PATH_MODEL): path of the file including model weights
 * -i (--FOLDER_DATA): folder where patches are stored
 * -l (--LIST_IMAGES): path of the csv file including the WSI IDs (used to create the input file with patch paths, in case it s missing.
 * -k (--keys): the size of the queue including examples.

-model.py: it includes the backbone of the architecture and the methods to implement MoCo and simCLR (including the weight updates and the projection layers).

-vision_transformer.py: in includes the methods to implement a Vision Transformer architecture.

## train_test_scripts
Scripts to train and test the networks on the classification of WSIs. The paper involves the classification of celiac disease (binary classification), lung cancer (multiclass classification), colon cancer (multilabel classification). Therefore, the methods to train and test involve the three types of classification. Furthermore, the methods include also different backbone architectures: CNNs and ViTs, all of the modeled as a Multiple Instance Learning (MIL) algorithm. The algorithms are: CLAM, transMIL, AB_MIL, AD_MIL, DSMIL and ViT (in model_transformers.py).

-train.py (all the training scripts show the same parameters):
 * -n (--N_EXP): number of experiment to run (in a k-fold cross validation scenario).
 * -c (--CNN): CNN backbone to use (only in non-ViT based architectures).
 * -e (--EPOCHS): number of epochs.
 * -m (--MAG): magnification to select.
 * -l (--LABELS): labels to use: GT (ground truth, manually made by experts) or SKET (automatically made using SKET tool).
 * -z (--hidden_space): size of the projection layers within the network.
 * -a (--algorithm): MIL algorithm to use (just for the CNNs).
 * -b (--batch_size): batch size to use (in case features are not pre-processed).
 * -i (--input_folder):'path folder input csv (there will be a train.csv file including IDs and labels).
 * -p (--output_folder): path folder where to store output model.
 * -p (--self_supervised): path folder with pretrained network.
 * -p (--weights): pre-trained weights to adopt, considering self-supervised algorithms.
 * -p (--TISSUE): tissue to use (Colon, Celiac, Lung). it influences the classes.
 * -p (--DATA_FOLDER): path of the folder where to patches are stored.
 * -f (--CSV_FOLDER): folder where csv including IDs and classes are stored.
 * --heads: number of heads for the ViT architecture.
 * --trans_layers: number of encoder layers for the ViT.
 * --max_seq_length: size of the input sequence for ViT.
 * --dropout: dropout value (only ViT).
 
-test.py (all the testing scripts show the same parameters):
 * -n (--N_EXP): number of experiment to run (in a k-fold cross validation scenario).
 * -c (--CNN): CNN backbone to use (only in non-ViT based architectures).
 * -d (--dataset): dataset to use for testing (in case more than one is available).
 * -e (--EPOCHS): number of epochs.
 * -m (--MAG): magnification to select.
 * -l (--LABELS): labels to use: GT (ground truth, manually made by experts) or SKET (automatically made using SKET tool).
 * -z (--hidden_space): size of the projection layers within the network.
 * -a (--algorithm): MIL algorithm to use (just for the CNNs).
 * -b (--batch_size): batch size to use (in case features are not pre-processed).
 * -i (--input_folder):'path folder input csv (there will be a train.csv file including IDs and labels).
 * -p (--output_folder): path folder where to store output model.
 * -p (--self_supervised): path folder with pretrained network.
 * -p (--weights): pre-trained weights to adopt, considering self-supervised algorithms.
 * -p (--TISSUE): tissue to use (Colon, Celiac, Lung). it influences the classes.
 * -p (--DATA_FOLDER): path of the folder where to patches are stored.
 * -f (--CSV_FOLDER): folder where csv including IDs and classes are stored.
 * --heads: number of heads for the ViT architecture.
 * --trans_layers: number of encoder layers for the ViT.
 * --max_seq_length: size of the input sequence for ViT.
 * --dropout: dropout value (only ViT).
 
## Acknoledgements
This project has received funding from the EuropeanUnion’s Horizon 2020 research and innovation programme under grant agree-ment No. 825292 [ExaMode](http://www.examode.eu). Infrastructure fromthe SURFsara HPC center was used to train the CNN models in parallel. 
