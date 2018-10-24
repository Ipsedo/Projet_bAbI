#!/usr/bin/env python
import model.fst_model as model
import torch as th
import data.data_processing

# test
taille_vocab = 15
taille_embedding = 9
taille_hidden_story = 21
taille_hidden_quest = 22

model = model.MyModel(taille_vocab, taille_embedding, taille_hidden_story, taille_hidden_quest)
print(model)

story1 = th.LongTensor([1, 5, 9, 4, 5, 14, 12, 6])
quest1 = th.LongTensor([5, 9, 13, 10])
ans = 10
inp = (story1, quest1, ans)

model.train()
out = model(inp)

print(out.shape)
