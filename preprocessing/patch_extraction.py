import sys, os
import openslide
from PIL import Image
import numpy as np
import pandas as pd 
from collections import Counter
from matplotlib import pyplot as plt
from skimage import io
import threading
import time
import collections
import time
from skimage import exposure
import json
import multiprocessing
import argparse

np.random.seed(0)

argv = sys.argv[1:]

parser = argparse.ArgumentParser(description='Configurations to train models.')
parser.add_argument('-c', '--USE_CASE', help='Colon, Celiac, Cervix',type=str, default='Colon')
parser.add_argument('-m', '--MAGNIFICATION', help='magnification level',type=int, default=10)
parser.add_argument('-t', '--THREADING', help='threading/processes',type=str, default='thread')
parser.add_argument('-i', '--INPUT_DATA', help='path of the csv including the WSI paths',type=str, default='')
parser.add_argument('-o', '--OUTPUT_PATH', help='path of output folder',type=str, default='')
parser.add_argument('-l', '--MASKS_PATH', help='path of the folder including tissue masks',type=str, default='')

args = parser.parse_args()

USE_CASE = args.USE_CASE
LIST_FILE = args.INPUT_DATA
PATH_OUTPUT = args.OUTPUT_PATH

MAGNIFICATION = args.MAGNIFICATION
MAGNIFICATION_str = str(MAGNIFICATION)

PATH_INPUT_MASKS = args.MASKS_PATH

PATH_OUTPUT = PATH_OUTPUT + '/' + USE_CASE + '/'
os.makedirs(PATH_OUTPUT, exist_ok = True)
PATH_OUTPUT = PATH_OUTPUT+'magnification_'+MAGNIFICATION_str+'x/'
os.makedirs(PATH_OUTPUT, exist_ok = True)


THREAD_NUMBER = 10
lockList = threading.Lock()
lockGeneralFile = threading.Lock()



def create_output_imgs(img,fname):
	#save file
	new_patch_size = 224
	img = img.resize((new_patch_size,new_patch_size))
	img = np.asarray(img)
	#io.imsave(fname, img)
	print("file " + str(fname) + " saved")

def check_background(glimpse,threshold,GLIMPSE_SIZE_SELECTED_LEVEL):
	b = False

	window_size = GLIMPSE_SIZE_SELECTED_LEVEL
	tot_pxl = window_size*window_size
	white_pxl = np.count_nonzero(glimpse)
	score = white_pxl/tot_pxl
	if (score>=threshold):
		b=True
	return b

def write_coords_local_file(fname,arrays):
		#select path
	output_dir = PATH_OUTPUT+'/'+fname+'/'
	filename_path = output_dir+fname+'_coords_densely.csv'
		#create file
	File = {'filename':arrays[0],'level':arrays[1],'x_top':arrays[2],'y_top':arrays[3]}
	df = pd.DataFrame(File,columns=['filename','level','x_top','y_top'])
		#save file
	np.savetxt(filename_path, df.values, fmt='%s',delimiter=',')

def write_paths_local_file(fname,listnames):
	output_dir = PATH_OUTPUT+'/'+fname+'/'
	filename_path = output_dir+fname+'_paths_densely.csv'
		#create file
	File = {'filenames':listnames}
	df = pd.DataFrame(File,columns=['filenames'])
		#save file
	np.savetxt(filename_path, df.values, fmt='%s',delimiter=',')

def eval_whitish_threshold(mask, thumb):
	a = np.ma.array(thumb, mask=np.logical_not(mask))
	mean_a = a.mean()

	THRESHOLD = 200
	if (mean_a<=155):
		THRESHOLD = 195.0
	elif (mean_a>155 and mean_a<=180):
		THRESHOLD = 200.0
	elif (mean_a>180):
		THRESHOLD = 205.0
	return THRESHOLD

def whitish_img(img, THRESHOLD_WHITE):
	b = True
	if (np.mean(img) > THRESHOLD_WHITE):
		b = False
	return b

def available_magnifications(mpp, level_downsamples):
	mpp = float(mpp)
	if (mpp<0.26):
		magnification = 40
	else:
		magnification = 20
	
	mags = []
	for l in level_downsamples:
		mags.append(magnification/l)
	
	return mags

def split_in_subpatches(img_np, patch_size):
	
	shape_x = img_np.shape[1]
	shape_y = img_np.shape[0]
	channels = 1
	
	n_patches_x = shape_x // patch_size
	n_patches_y = shape_y // patch_size
	
	left_x = shape_x % patch_size
	left_y = shape_y % patch_size
	
	img_np = img_np[:-left_y, : -left_x]
	
	shape_x = img_np.shape[1]
	shape_y = img_np.shape[0]
	
	assert shape_x % patch_size == 0
	assert shape_y % patch_size == 0
	
	n_patches = int(n_patches_x * n_patches_y)

	#img_np = img_np.swapaxes(0,2)
	print(img_np.shape)
	#img_reshaped = np.reshape(img_np, (n_patches, patch_size, patch_size, 3)).swapaxes(1,2)
	img_reshaped = img_np.reshape(
										n_patches_y, patch_size,
										n_patches_x, patch_size
									).swapaxes(1,2)
	
	
	img_reshaped = img_reshaped.reshape(n_patches, patch_size, patch_size)
	
	return img_reshaped, n_patches_x, n_patches_y
#estrae glimpse e salva metadati relativi al glimpse
def analyze_file(filename):
	global filename_list_general, MAGNIFICATION

	patches = []
	print(filename)
	new_patch_size = 224
	print(filename, 'to open')
	#file = openslide.open_slide(filename)

	try:
		file = openslide.OpenSlide(filename)
		print(filename, 'loaded')
		flag_file = True
	except:
		try:
			file = openslide.open_slide(filename)
			print(filename, 'loaded')
			flag_file = True
		except:
			flag_file = False

	try:
		mpp = file.properties['openslide.mpp-x']
	except:
		mpp = 0.24

	if (flag_file):

		level_downsamples = file.level_downsamples
		mags = available_magnifications(mpp, level_downsamples)

		level = 0
			#load file
		

			#load mask
		fname = os.path.split(filename)[-1]
			#check if exists
		fname_mask = PATH_INPUT_MASKS+fname+'/'+fname+'_mask_use.png' 


		array_dict = []

			#levels for the conversion
		WANTED_LEVEL = MAGNIFICATION
		MASK_LEVEL = 1.25
		HIGHEST_LEVEL = mags[0]
		#AVAILABLE_LEVEL = select_nearest_magnification(WANTED_LEVEL, mags, level_downsamples)
		
		RATIO_WANTED_MASK = WANTED_LEVEL/MASK_LEVEL
		RATIO_HIGHEST_MASK = HIGHEST_LEVEL/MASK_LEVEL

		WINDOW_WANTED_LEVEL = new_patch_size

		GLIMPSE_SIZE_SELECTED_LEVEL = WINDOW_WANTED_LEVEL

		GLIMPSE_SIZE_MASK = np.around(GLIMPSE_SIZE_SELECTED_LEVEL/RATIO_WANTED_MASK)
		GLIMPSE_SIZE_MASK = int(GLIMPSE_SIZE_MASK)

		GLIMPSE_HIGHEST_LEVEL = np.around(GLIMPSE_SIZE_MASK*RATIO_HIGHEST_MASK)
		GLIMPSE_HIGHEST_LEVEL = int(GLIMPSE_HIGHEST_LEVEL)
		
		STRIDE_SIZE_MASK = 0
		TILE_SIZE_MASK = GLIMPSE_SIZE_MASK+STRIDE_SIZE_MASK

		PIXEL_THRESH = 0.5

		fname_out = PATH_OUTPUT+'/'+fname+'/'+fname+'_coords_densely.csv'
		flag_csv = False

		try:
			local_csv = pd.read_csv(fname_out, sep=',', header=None).values

			if(len(local_csv)>1):
				flag_csv = True
				print(fname, 'exists')

		except:
			pass

		#print(fname_out)
		#flag_csv = False
		n_image = 0
		threshold = PIXEL_THRESH
		output_dir = PATH_OUTPUT+fname

		#if (os.path.isfile(fname_mask) and os.path.exists(fname_out)==False):
		if (os.path.isfile(fname_mask) and flag_csv==False):
		#if (os.path.isfile(fname_mask)):
				#creates directory
			#output_dir = PATH_OUTPUT+fname
			os.makedirs(output_dir,exist_ok = True)

				#create CSV file structure (local)
			filename_list = []
			level_list = []
			x_list = []
			y_list = []

			img = Image.open(fname_mask)

			thumb = file.get_thumbnail(img.size)
			thumb = thumb.resize(img.size)
			mask_np = np.asarray(thumb)
			img = np.asarray(img)

			mask_3d = np.repeat(img[:, :, np.newaxis], 3, axis=2)
			
			WHITISH_THRESHOLD = eval_whitish_threshold(mask_3d, mask_np)

			mask_np = np.asarray(img)

			cont = 0
			
			threshold_iterations = 5000000
			a = True
			tot_patches = 0
			
			try:
				patches_from_mask, n_patches_x, n_patches_y = split_in_subpatches(mask_np, GLIMPSE_SIZE_MASK)
				tot_patches = patches_from_mask.shape[0]
			except Exception as e:
				print(e)
				a = False

			cont_p = 0
			x_ini = 0
			y_ini = 0  

			n_image = 0
			threshold = PIXEL_THRESH


			while(cont_p<tot_patches and a==True):
				
				glimpse = patches_from_mask[cont_p]

				check_flag = check_background(glimpse,threshold,TILE_SIZE_MASK)

				x_ini = int((cont_p % n_patches_x) * TILE_SIZE_MASK)
		
				if (x_ini == 0 and cont_p > 1 ):
					y_ini = y_ini + TILE_SIZE_MASK
					x_ini = 0

				if(check_flag):
					#print(x_ini, y_ini)

					fname_patch = output_dir+'/'+fname+'_'+str(n_image)+'.png'
						#change to magnification 40x
					x_coords_0 = int(x_ini*RATIO_HIGHEST_MASK)
					y_coords_0 = int(y_ini*RATIO_HIGHEST_MASK)

					file_40x = file.read_region((x_coords_0,y_coords_0),level,(GLIMPSE_HIGHEST_LEVEL,GLIMPSE_HIGHEST_LEVEL))
					file_40x = file_40x.convert("RGB")

					new_patch_size = 224
					save_im = file_40x.resize((new_patch_size,new_patch_size))
					save_im = np.asarray(save_im)	

					bool_white = whitish_img(save_im,WHITISH_THRESHOLD)
					#bool_white = True
					bool_exposure = exposure.is_low_contrast(save_im)
				
					if (bool_white):
						if bool_exposure==False:
						#if (exposure.is_low_contrast(save_im)==False):

							io.imsave(fname_patch, save_im)

							#add to arrays (local)
							filename_list.append(fname_patch)
							level_list.append(level)
							x_list.append(x_coords_0)
							y_list.append(y_coords_0)
							n_image = n_image+1
							#save the image
							#create_output_imgs(file_10x,fname)
						"""
						else:
							print("low_contrast " + str(output_dir))
						"""
				
				cont_p = cont_p + 1


				#add to general arrays
			if (n_image!=0):
				#lockGeneralFile.acquire()
				#filename_list_general.append(output_dir)

				print("WSI done: " + filename)
				#print("len filename " + str(len(filename_list_general)) + "; WSI done: " + filename)
				print("extracted " + str(n_image) + " patches")
				#lockGeneralFile.release()
				write_coords_local_file(fname,[filename_list,level_list,x_list,y_list])
				write_paths_local_file(fname,filename_list)
			else:
				print("ZERO OCCURRENCIES " + str(output_dir))

		file.close()

def explore_list(list_dirs):
	global list_dicts, n, csv_binary, csv_multiclass
	#print(threadname + str(" started"))

	for i in range(len(list_dirs)):
		analyze_file(list_dirs[i])
	#print(threadname + str(" finished"))

def chunker_list(seq, size):
		return (seq[i::size] for i in range(size))

def main():
	#create output dir if not exists
	start_time = time.time()
	global list_dicts, n, filename_list_general

	n = 0
		#create dir output
	if not os.path.exists(PATH_OUTPUT):
		print("create_output " + str(PATH_OUTPUT))
		os.makedirs(PATH_OUTPUT)

	list_dirs = pd.read_csv(LIST_FILE, sep=',', header=None).values.flatten()
	
		#split in chunks for the threads
	list_dirs = list(chunker_list(list_dirs,THREAD_NUMBER))
	print(len(list_dirs))

	threads = []
	for i in range(THREAD_NUMBER):
		#t = multiprocessing.Process(target=explore_list,args=([list_dirs[i]]))
		t = threading.Thread(target=explore_list,args=([list_dirs[i]]))
		threads.append(t)

	for t in threads:
		t.start()
		#time.sleep(60)

	for t in threads:
		t.join()
	
		#prepare data
	
	elapsed_time = time.time() - start_time
	print("elapsed time " + str(elapsed_time))


if __name__ == "__main__":
	main()
