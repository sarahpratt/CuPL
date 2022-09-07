import numpy as np
import torch
import clip
from pkg_resources import packaging
from imagenet_prompts.standard_image_prompts import imagenet_templates
import pdb
from collections import defaultdict
from imagenetdataset import ImagenetDataset
from PIL import Image
import PIL
import json
from tqdm import tqdm

PATH_TO_IMAGENET = "../val"
PATH_TO_PROMPTS = "./imagenet_prompts/CuPL_image_prompts.json"

model, preprocess = clip.load("ViT-L/14")
model.eval()

all_images = ImagenetDataset(PATH_TO_IMAGENET, transform=preprocess)
loader = torch.utils.data.DataLoader(all_images, batch_size=512, num_workers=8)

def zeroshot_classifier(classnames, textnames, templates):
	with torch.no_grad():
		zeroshot_weights = []
		i = 0
		for classname in tqdm(classnames):
			texts = [template.format(textnames[i]) for template in templates] #format with class
			texts = clip.tokenize(texts).cuda() #tokenize
			class_embeddings = model.encode_text(texts) #embed with text encoder
			class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
			class_embedding = class_embeddings.mean(dim=0)
			class_embedding /= class_embedding.norm()
			zeroshot_weights.append(class_embedding)
			i += 1
		zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
	return zeroshot_weights

def zeroshot_classifier_gpt(classnames, textnames, templates, use_both):
	with open(PATH_TO_PROMPTS) as f:
		gpt3_prompts = json.load(f)

	with torch.no_grad():
		zeroshot_weights = []
		i = 0
		for classname in tqdm(classnames):
			if use_both:
				texts = [template.format(textnames[i]) for template in templates]
			else:
				texts = []

			for t in gpt3_prompts[textnames[i]]:
				texts.append(t)
			texts = clip.tokenize(texts, truncate=True).cuda() #tokenize
			class_embeddings = model.encode_text(texts) #embed with text encoder
			class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
			class_embedding = class_embeddings.mean(dim=0)
			class_embedding /= class_embedding.norm()
			zeroshot_weights.append(class_embedding)
			i += 1

		zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
	return zeroshot_weights

print("\nCreating standard text embeddings...")
zeroshot_weights_base = zeroshot_classifier(all_images.idx_to_label, all_images.idx_to_text, imagenet_templates)
print("Done.\n")

print("Creating CuPL text embeddings...")
zeroshot_weights_cupl = zeroshot_classifier_gpt(all_images.idx_to_label, all_images.idx_to_text, imagenet_templates, False)
print("Done.\n")

print("Creating combined text embeddings...")
zeroshot_weights_gpt_both = zeroshot_classifier_gpt(all_images.idx_to_label, all_images.idx_to_text, imagenet_templates, True)
print("Done.\n")


total = 0.
correct_base = 0.
correct_cupl = 0.
correct_both = 0.

print("Classifying ImageNet...")

with torch.no_grad():

	for i, (images, target, num) in enumerate(tqdm(loader)):
		images = images.cuda()
		target = target.cuda()

		# predict
		image_features = model.encode_image(images)
		image_features /= image_features.norm(dim=-1, keepdim=True)

		logits_base = image_features @ zeroshot_weights_base
		logits_cupl = image_features @ zeroshot_weights_cupl
		logits_both = image_features @ zeroshot_weights_gpt_both

		pred_base = torch.argmax(logits_base, dim =1)
		pred_cupl = torch.argmax(logits_cupl, dim =1)
		pred_both = torch.argmax(logits_both, dim =1)

		for j in range(len(target)):
			total += 1.
			if pred_base[j] == target[j]:
				correct_base += 1.
			if pred_cupl[j] == target[j]:
				correct_cupl += 1.
			if pred_both[j] == target[j]:
				correct_both += 1.


print()
top1 = (correct_base / total) * 100
print(f"Top-1 accuracy standard: {top1:.2f}")

top1 = (correct_cupl / total) * 100
print(f"Top-1 accuracy CuPL: {top1:.2f}")

top1 = (correct_both / total) * 100
print(f"Top-1 accuracy both: {top1:.2f}")
