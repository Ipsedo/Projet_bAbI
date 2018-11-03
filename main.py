#!/usr/bin/env python
import model.snd_model as mod
import torch as th
import data.data_processing as data_process
import data.data_processingqa20 as data_process20

# Données de train
#f = open("./res/tasks_1-20_v1-2/en/qa20_agents-motivations_train.txt","r")
#data_train = data_process20.process_data(data_process20.split_file(f))
#data_train = data_process20.split_sentence(data_train)
#vocab_train, data_train = data_process20.make_vocab_and_transform_data(data_train)
f = open("./res/tasks_1-20_v1-2/en/qa1_single-supporting-fact_train.txt", "r")
data_train = data_process.process_data(data_process.split_file(f))
data_train = data_process.split_sentence(data_train)
vocab_train, data_train = data_process.make_vocab_and_transform_data(data_train)
f.close()

# Données de test
#f = open("./res/tasks_1-20_v1-2/en/qa20_agents-motivations_test.txt","r")
#data_test = data_process20.process_data(data_process20.split_file(f))
#data_test = data_process20.split_sentence(data_test)
#data_test = data_process20.make_data_with_vocab(data_test, vocab_train)
f = open("./res/tasks_1-20_v1-2/en/qa1_single-supporting-fact_test.txt", "r")
data_test = data_process.process_data(data_process.split_file(f))
data_test = data_process.split_sentence(data_test)
data_test = data_process.make_data_with_vocab(data_test, vocab_train)
f.close()

taille_vocab=len(vocab_train)
taille_embedding = 50
taille_hidden_story = 100
taille_hidden_quest = 100

#story_max_len, quest_max_len = data_process.get_story_quest_max_len(data_train)

model = mod.SndModel(taille_vocab, taille_embedding, taille_hidden_story, taille_hidden_quest)
loss_fn = th.nn.NLLLoss()

if data_process.use_cuda():
	model.cuda()
	loss_fn.cuda()

optimizer = th.optim.Adam(model.parameters(), lr=1e-3)

nbEpoch = 20


for e in range(nbEpoch):
	model.train()
	nb_train = 0
	epoch_loss = 0
	for chall in data_train:
		model.zero_grad()
		out = model(chall)
		target = th.LongTensor([chall[2]])
		if data_process.use_cuda():
			target = target.cuda()
		loss = loss_fn(out.unsqueeze(0), target)
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
