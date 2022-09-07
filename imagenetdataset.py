import pathlib
import tarfile
import requests
import shutil
from collections import defaultdict

from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
import os
from imagenet_classnames.sense_to_idx import sense
from imagenet_classnames.imagenet_classes import imagenet_classes
import pdb


class ImagenetDataset(Dataset):
	def __init__(self, path, transform=None):
		self.transform = transform
		self.paths = []
		self.labels = []
		self.label_to_idx = defaultdict(int)
		self.idx_to_label = []
		self.idx_to_text = []


		sense_to_name = {}
		i = 0
		for item in sense:
			sense_num = "n" + sense[item]['id'].split("-")[0]
			sense_to_name[sense_num] = imagenet_classes[i]
			i += 1

		for directory in os.listdir(path):
			First = True
			f = os.path.join(path, directory)
			if os.path.isdir(f):
				for image in os.listdir(f):
					ext = image.split('.')[-1]
					if ext == 'JPEG':
						image_path = os.path.join(f, image)
						self.paths.append(image_path)
						label_name = sense_to_name[directory]
						if First:
							First = False
							self.label_to_idx[directory] = len(self.idx_to_label)
							self.idx_to_label.append(directory)
							self.idx_to_text.append(label_name)
						self.labels.append(self.label_to_idx[directory])

	def __len__(self):
		return len(self.paths)

	def __getitem__(self, i):
		img, label = Image.open(self.paths[i]), int(self.labels[i])
		if self.transform is not None:
			img = self.transform(img)
		return img, label, i
