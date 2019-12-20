import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import random
import json
import torch
import torch.nn.functional as F
import torch.nn as nn
import os 

def load_preprocess(file_path = "data/training_data.json"):
	stemmer = LancasterStemmer()

	# 1. Load data
	words = []
	labels = []
	docs_x = []
	docs_y = []
	with open(file_path) as file:
	    data = json.load(file)
	    for intent in data['intents']:
	      for pattern in intent['patterns']:
	        wrds = nltk.word_tokenize(pattern)
	        words.extend(wrds)
	        docs_x.append(wrds)
	        docs_y.append(intent['tag'])
	      if intent['tag'] not in labels:
	        labels.append(intent['tag'])

	# 2. Pre_process data

	# stem & sort the words list
	words = [stemmer.stem(w.lower()) for w in words if w != "?"]
	words = sorted(list(set(words)))
	# sort the labels
	labels = sorted(labels)
	# print('labels: ', labels)
	# print('words: ', words)

	features = []
	target = []


	for i, doc in enumerate(docs_x):
	  sub_features = []
	  wrds = [stemmer.stem(w) for w in doc]
	  for w in words:
	    if w in wrds:
	      sub_features.append(1)
	    else:
	      sub_features.append(0)
	  features.append(sub_features)
	  
	  col = labels.index(docs_y[i])
	  target.append(col)

	features = torch.from_numpy(np.array(features)).type(torch.FloatTensor)
	target = torch.from_numpy(np.array(target)).type(torch.LongTensor)

	return features, target, words, docs_x, docs_y, labels, data

# 3. Define Model & training
class DNNModel(nn.Module):
  def __init__(self, inp_dim, out_dim):
    super().__init__()
    self.fc1 = nn.Linear(inp_dim, 10)
    self.fc2 = nn.Linear(10, 10)
    self.fc3 = nn.Linear(10, out_features=out_dim)
  def forward(self, inp):
    out = self.fc1(inp)
    out = F.relu(out)
    out = self.fc2(out)
    out = torch.tanh(out)
    out = self.fc2(out)
    out = F.elu(out)
    return out

def train(data_path = 'data/training_data.json'):
	features, target, words, docs_x, docs_y, labels, data = load_preprocess(data_path)
	model = DNNModel(features.size()[1], 6)
	optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
	loss_fn = nn.CrossEntropyLoss()
	model.train()
	for i in range(4000):
		predicted = model(features)

		optimizer.zero_grad()
		loss = loss_fn(predicted, target)
		loss.backward()
		optimizer.step()

		if(i%200 == 0):
			prediction = torch.max(predicted, 1)[1]
			correct = (prediction == target).sum()
			accuracy = correct/float(target.size()[0])
			print("iter = {}, loss = {}, accuracy = {}".format(i, loss, accuracy))
	torch.save(model, 'mymodel.pt')




# processing input for output
def bag_of_words(sentence, words):
	stemmer = LancasterStemmer()
	bag = [0 for i in range(len(words))]
	wrd_extract = nltk.word_tokenize(sentence)
	wrd_extract = [stemmer.stem(w.lower()) for w in wrd_extract if w != "?"]

	for i, w in enumerate(words):
		if w in wrd_extract:
			bag[i] = 1
	return torch.from_numpy(np.array(bag)).type(torch.FloatTensor)

def response(inp, model, data_path = 'data/training_data.json'):
	features, target, words, docs_x, docs_y, labels, data = load_preprocess(data_path)

	output = model(bag_of_words(inp, words))
	out_max_point = torch.max(F.softmax(output), 0)[0].data
	if( out_max_point < 0.5):
	 	return "Sorry. I don't understand. Ask another question !"
	else:
		response_index = torch.max(output, 0)[1].data
		response = labels[response_index]
		for i, res in enumerate(data["intents"]):
			if res['tag'] == response:
				responses = res['responses']
		return random.choice(responses)
# train()