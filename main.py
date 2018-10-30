#!/usr/bin/env python
import model.fst_model as mod
import torch as th

import data.data_processing as data_process
import data.data_processingqa20 as data_process20

# Données de train
# f = open("./res/tasks_1-20_v1-2/en/qa1_single-supporting-fact_train.txt","r")
f = open("./res/tasks_1-20_v1-2/en/qa20_agents-motivations_train.txt","r")
data_train = data_process20.process_data(data_process20.split_file(f))
data_train = data_process20.split_sentence(data_train)
vocab_train, data_train = data_process20.make_vocab_and_transform_data(data_train)
f.close()

# Données de test
# f = open("./res/tasks_1-20_v1-2/en/qa1_single-supporting-fact_test.txt","r")
f = open("./res/tasks_1-20_v1-2/en/qa20_agents-motivations_test.txt","r")
data_test = data_process20.process_data(data_process20.split_file(f))
data_test = data_process20.split_sentence(data_test)
data_test = data_process20.make_data_with_vocab(data_test, vocab_train)
f.close()

taille_vocab=len(vocab_train)
taille_embedding = 10
taille_hidden_story = 5
taille_hidden_quest = 7

model = mod.MyModel(taille_vocab, taille_embedding, taille_hidden_story, taille_hidden_quest)
optimizer = th.optim.Adagrad(model.parameters(), lr=1e-2)
loss_fn = th.nn.CrossEntropyLoss()

nbEpoch = 20

for e in range(nbEpoch):
	model.train()
	nb_train = 0
	epoch_loss = 0
	for chall in data_train:
		model.zero_grad()
		out = model(chall)
		loss = loss_fn(out.unsqueeze(0), th.LongTensor([chall[2]]))
		loss.backward()
		optimizer.step()
		nb_train += 1
		epoch_loss += loss.item()

	nb_test = 0
	nb_err = 0
	model.eval()
	for chall in data_test:
		out = model(chall)
		nb_err += 1 if out.argmax(0) != chall[2] else 0
		nb_test += 1

	print("Epoch (%d) : err_rate = %f, loss = %f" % (e, nb_err / nb_test, epoch_loss / nb_train))


model.train()
out=model(data_train[0])
print(out)
print(len(vocab_train))

s1=data_train[50][0].numpy()
print(type(s1))
str=""
inv_map = {v: k for k, v in vocab_train.items()}
for d in s1:
	str += inv_map[d]
print(str)