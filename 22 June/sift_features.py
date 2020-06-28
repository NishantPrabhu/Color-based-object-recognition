
# Dependencies
import os 
import cv2
import pickle 
from glob import glob
from tqdm import tqdm 

# Create sift features
sift = cv2.xfeatures2d.SIFT_create()
file_desc_map = {}

folders = os.listdir('../../data')
folders.remove('test_images')
progress = tqdm(total=len(folders), position=0, desc='Progress')
folder_name = tqdm(total=0, position=1, bar_format='{desc}') 

for fol in folders:

	folder_name.set_description_str(f"Now processing {fol} ...")
	files = glob('../../data/'+fol+'/*.jpg')
	folder_pbar = tqdm(total=len(files), position=2, desc='Folder', leave=False)
	
	for path in files:
		img = cv2.imread(path, 0)
		_, desc = sift.detectAndCompute(img, None)
		file_desc_map.update({path: desc})
		folder_pbar.update(1)
	
	progress.update(1)


# Save SIFT features
with open('../../saved_data/22 Jun/file_sift_map.pkl', 'wb') as f:
	pickle.dump(file_desc_map, f)