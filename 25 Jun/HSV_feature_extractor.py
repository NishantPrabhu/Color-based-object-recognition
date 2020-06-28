
# Read all images in LAB color space
# Extract top colors using MiniBatchKMeans
# LAB better represents human method of perceiving colors 

# Dependencies
import os
import cv2
import pickle
import numpy as np
from tqdm import tqdm
from ntpath import basename
from sklearn.cluster import MiniBatchKMeans
from skimage.color import rgb2lab, deltaE_cie76
from warnings import filterwarnings
filterwarnings(action='ignore')

	
# Function to create color_ratio dict
def get_color_ratio(img_path):
	
	img = cv2.imread(img_path)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	x = img.reshape(img.shape[0]*img.shape[1], -1)
	kmeans = MiniBatchKMeans(n_clusters=5)
	kmeans.fit(x)
	centers, labels = kmeans.cluster_centers_, np.asarray(kmeans.labels_)
	
	color_ratio = {}
	for i, (c, l) in enumerate(zip(centers, np.unique(labels))):
		ratio = len(np.where(labels == l)[0])/len(labels)
		color_ratio.update({i: (c, ratio)})
		
	return color_ratio


# Color matching function
def match_color(color_1, ratio_1, color_2, ratio_2, thresh=60):
	
	try:
		color_1 = color_1.tolist()
		color_2 = color_2.tolist()
	except:
		raise ValueError("Bad format for color; provide as numpy arrays")
		
	color_1 = rgb2lab(np.uint8(np.asarray([[color_1]])))
	color_2 = rgb2lab(np.uint8(np.asarray([[color_2]])))
	diff = deltaE_cie76(color_1, color_2)
	
	if diff < thresh:
		if min(ratio_1, ratio_2)/max(ratio_1, ratio_2) > 0.7:
			return 1
	
	return 0


# Image matching function
def match_image_color_dict(color_dict_1, color_dict_2):
	matches = 0
	for i, (c_1, r_1) in color_dict_1.items():
		for i, (c_2, r_2) in color_dict_2.items():
			matches += match_color(c_1, r_1, c_2, r_2, thresh=60)
	
	return matches
	
	
	
if __name__ == "__main__":
	
	# Extract all paths to a list
	folders = os.listdir("../../data")
	folders.remove("test_images")
	all_paths = []

	for fol in folders:
		files = os.listdir("../../data/"+fol)
		paths = ["../../data/"+fol+"/"+f for f in files]
		all_paths.extend(paths)
		
	# Using fewer paths for testing
	all_paths = all_paths[:2000]
		
	# Read each image, compute color ratios and store values in dict
	"""file_color_dict = {}
	color_progress = tqdm(total=len(all_paths), position=0, desc='Progress', leave=False)
	
	for path in all_paths:
		color_ratio = get_color_ratio(path)	
		file_color_dict.update({path: color_ratio})
		color_progress.update(1)
		

	# Save this dictionary
	with open("../../saved_data/25 Jun/file_color_dict.pkl", "wb") as f:
		pickle.dump(file_color_dict, f)"""
	
	
	with open("../../saved_data/25 Jun/file_color_dict.pkl", "rb") as f:
		file_color_dict = pickle.load(f)
	
	# Evaluate the model
	progress = tqdm(total=len(all_paths), position=0, desc='Progress')
	acc_status = tqdm(total=0, position=1, bar_format='{desc}')
	acc3_status = tqdm(total=0, position=2, bar_format='{desc}')
	
	correct, correct3 = 0, 0
	
	for i, trg_path in enumerate(all_paths):
		
		scores = {}
		trg_color_dict = get_color_ratio(trg_path)
		
		for path, color_dict in file_color_dict.items():
			matches = match_image_color_dict(trg_color_dict, color_dict)
			scores.update({path: matches})
			
		sorted_paths = sorted(list(scores.keys()), key=lambda x: scores[x], reverse=True)
		
		if basename(trg_path) == basename(sorted_paths[0]):
			correct += 1
			
		if basename(trg_path) in [basename(p) for p in sorted_paths[:3]]:
			correct3 += 1
			
		progress.update(1)
	
		if i % 10 == 0:
			acc_status.set_description_str("Accuracy: {:.2f}%".format(
				100. * correct / (i+1)
			))
			acc3_status.set_description_str("Top 3 accuracy: {:.2f}%".format(
				100. * correct3 / (i+1)
			))
			
	
	# Final accuracy after everything is done
	print("\n\n\n\n\n")
	print("Final accuracy: {:.2f}%".format(100. * correct / len(all_paths)))
	print("Final top 3 accuracy: {:.2f}%".format(100. * correct3 / len(all_paths)))
	
	
	
	
	
	
	

	