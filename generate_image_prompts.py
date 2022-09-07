import os
import openai
import json
import pdb
from imagenet_classnames.imagenet_classes import imagenet_classes
from tqdm import tqdm

openai.api_key = "INSERT YOUR OPENAI BETA API KEY"
json_name = "json_name.json"

category_list = imagenet_classes
all_responses = {}
vowel_list = ['A', 'E', 'I', 'O', 'U']

for category in tqdm(category_list):

	if category[0] in vowel_list:
		article = "an"
	else:
		article = "a"

	prompts = []
	prompts.append("Describe what " + article + " " + category + " looks like")
	prompts.append("How can you identify " + article + " " + category + "?")
	prompts.append("What does " + article + " " + category + " look like?")
	prompts.append("Describe an image from the internet of " + article + " "  + category)
	prompts.append("A caption of an image of "  + article + " "  + category + ":")

	all_result = []
	for curr_prompt in prompts:
		response = openai.Completion.create(
		    engine="text-davinci-002",
		    prompt=curr_prompt,
		    temperature=.99,
			max_tokens = 50,
			n=10,
			stop="."
		)

		for r in range(len(response["choices"])):
			result = response["choices"][r]["text"]
			all_result.append(result.replace("\n\n", "") + ".")

	all_responses[category] = all_result

with open(json_name, 'w') as f:
	json.dump(all_responses, f, indent=4)
