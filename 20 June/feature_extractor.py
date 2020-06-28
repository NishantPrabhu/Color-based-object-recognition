# Dependencies
import os
import cv2
import pickle
import numpy as np
from glob import glob
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans
from tqdm import tqdm


# Function to obtain all colors
def get_all_colors(root_dir, n_colors=10):
	
	"""Find colors in each image and collect all colors"""
	
	all_colors = []; file_color_map = {}
	folders = os.listdir(root_dir)
	
	total_progress = tqdm(total=len(folders), position=0, desc='Progress')
	folder_display = tqdm(total=0, position=1, bar_format='{desc}')
	
	for folder in folders:
		folder_display.set_description_str(f"Now processing {folder} ...")
		files = glob(root_dir+'/'+folder+'/*.jpg')
		folder_progress = tqdm(total=len(files), position=2, desc='Folder', leave=False)
		
		for path in files:
			img = cv2.imread(path)
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			img = img.reshape(img.shape[0]*img.shape[1], -1)
			clt = MiniBatchKMeans(n_clusters=n_colors)
			clt.fit(img)
			all_colors.extend(clt.cluster_centers_.round())
			file_color_map.update({path: clt.cluster_centers_.round()})
			folder_progress.update(1)
		
		total_progress.update(1)
		
	return np.array(all_colors), file_color_map


# Generate the files and save them
all_colors, file_color_map = get_all_colors("../../data", n_colors=10)

with open("../../saved_data/20 Jun/all_colors.pkl", "wb") as f:
	pickle.dump(all_colors, f)
	
with open("../../saved_data/20 Jun/file_color_map.pkl", "wb") as f:
	pickle.dump(file_color_map, f)
	
	
# Perform KMeans on centers
kmeans_clt = KMeans(n_clusters=10)
kmeans_clt.fit(all_colors)
centers, labels = kmeans_clt.cluster_centers_, kmeans_clt.labels_


def color_uid(rgb):
	return ''.join([str(int(i)) for i in rgb])


# Color label map and file_color_map
uids = [color_uid(i) for i in all_colors]
color_label_map = dict(zip(uids, labels))

for path, colors in file_color_map.items():
	labels = [color_label_map[color_uid(i)] for i in colors]
	file_color_map.update({path: labels})
	

# Save these objects
with open("../../saved_data/20 Jun/file_color_map.pkl", "wb") as f:
	pickle.dump(file_color_map, f)
	
with open("../../saved_data/20 Jun/kmeans_clt.pkl", "wb") as f:
	pickle.dump(kmeans_clt, f)






			