#!/usr/bin/env python
import model.fst_model as mod
import torch as th
import data.data_processing as data_process


# test donnée
f= open("./res/tasks_1-20_v1-2/en/qa1_single-supporting-fact_train.txt","r")
data_train = data_process.process_data(data_process.split_file(f))
data=data_process.split_sentence(data_train)
vocab, prepared_data = data_process.make_vocab_and_transform_data(data)

taille_vocab=len(vocab)
taille_embedding = 9
taille_hidden_story = 30
taille_hidden_quest = 60

model = mod.MyModel(taille_vocab, taille_embedding, taille_hidden_story, taille_hidden_quest)
model.train()
out=model(prepared_data[0])
print(out)
print(len(vocab))

s1=prepared_data[50][0].numpy()
print(type(s1))
str=""
inv_map = {v: k for k, v in vocab.items()}
for d in s1:
	str += inv_map[d]
print(str)