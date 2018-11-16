import torch as th
import torch.nn as nn


class MyModel(nn.Module):
	def __init__(self, vocab_size, embedding_size, story_hidden_size, quest_hidden_size):
		"""

		:param vocab_size: taille du vocabulaire
		:param embedding_size: taille de la sortie d'embedding
		:param story_hidden_size: taille du vecteur cachée pour l'histoire
		:param quest_hidden_size: taille du vecteur cachée pour la question
		"""
		super(MyModel, self).__init__()

		self.vocab_size = vocab_size
		self.embedding_size = embedding_size
		self.story_hidden_size = story_hidden_size
		self.quest_hidden_size = quest_hidden_size

		# Une couche d'embedding et un réseau récurrent pour l'histoire
		self.embedding_story = nn.Embedding(self.vocab_size, self.embedding_size)
		self.rnn_story = nn.RNN(self.embedding_size, self.story_hidden_size)

		# De même pour la question
		self.embedding_quest = nn.Embedding(self.vocab_size, self.embedding_size)
		self.rnn_quest = nn.RNN(self.embedding_size, self.quest_hidden_size)


		# Puis une couche lineaire qui prendra la concaténation de la dernière sortie
		# des deux réseaux récurrent (histoire et question), et rendra autant de prédictions
		# qu'il y a de mots dans le vocabulaire
		self.lin = nn.Linear(self.story_hidden_size + self.quest_hidden_size, self.vocab_size)
		# Fonction d'activation
		self.act = nn.LogSoftmax(dim=0)

	def forward(self, inp):
		"""
		input = tuple(story, quest, reponse)
		story = sac d'indice de mot
			Pour la premiere question, on prend les 2 1eres affirmation
			Pour la 2e : on prend les 4 1eres affirmation
			etc.
		quest = sac de mot
		:param inp: tuple(story, quest, reponse)
		:return: sortie du forward
		"""
		story, quest, _ = inp
		# story.size=(nb_mot_story)
		# quest.size=(nb_mot_quest)

		outStory = self.embedding_story(story)
		# outStory.size = (nb_mot_story, taille_embedding)
		outStory, _ = self.rnn_story(outStory.unsqueeze(1), th.zeros(1, 1, self.story_hidden_size))
		# outStory.size = (nb_mot, 1, story_hidden_size)

		outQuest = self.embedding_quest(quest)
		outQuest, _ = self.rnn_quest(outQuest.unsqueeze(1), th.zeros(1, 1, self.quest_hidden_size))

		outStory = outStory.squeeze(1)
		# outStory.size = (nb_mot, story_hidden_size)
		outQuest = outQuest.squeeze(1)

		outStory = outStory[-1]  # get last element
		outQuest = outQuest[-1]  # get last element

		out = th.cat((outStory, outQuest))
		out = self.lin(out)
		return self.act(out)


class MyModel2(nn.Module):
	def __init__(self, vocab_size, embedding_size, hidden_size):
		"""

		:param vocab_size: taille du vocabulaire
		:param embedding_size: taille d'embedding
		:param hidden_size:
		"""
		super(MyModel2, self).__init__()

		self.vocab_size = vocab_size
		self.embedding_size = embedding_size
		self.hidden_size = hidden_size

		# Premier vecteur caché du réseau récurrent pour la question et l'histoire
		self.fst_hidden_story = th.randn(1, 1, self.hidden_size)
		self.fst_hidden_quest = th.randn(1, 1, self.hidden_size)

		# Une seule couche d'embedding
		self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)

		# Un seul réseau récurrent
		self.rnn = nn.RNN(self.embedding_size, self.hidden_size)

		# Une couche linéaire prenant la dernière sortie du réseau récurrent
		# pour l'histoire et la question et renvoyant self.vocab_size predictions
		self.lin = nn.Linear(2 * self.hidden_size, self.vocab_size)
		self.act = nn.LogSoftmax(dim=0)

	def forward(self, inp):
		"""
		input = tuple(story, quest, reponse)
		story = sac d'indice de mot
			Pour la premiere question, on prend les 2 1eres affirmation
			Pour la 2e : on prend les 4 1eres affirmation
			etc.
		quest = sac de mot
		:param inp:
		:return:
		"""
		story, quest, _ = inp
		# story.size=(nb_mot_story)
		# quest.size=(nb_mot_quest)

		outStory = self.embedding(story)
		# outStory.size = (nb_mot_story, taille_embedding)
		outStory, _ = self.rnn(outStory.unsqueeze(1), self.fst_hidden_story)
		# outStory.size = (nb_mot, 1, story_hidden_size)

		outQuest = self.embedding(quest)
		outQuest, _ = self.rnn(outQuest.unsqueeze(1), self.fst_hidden_quest)

		outStory = outStory.squeeze(1) # outStory.size = (nb_mot, story_hidden_size)
		outQuest = outQuest.squeeze(1)

		outStory = outStory[-1]  # get last element
		outQuest = outQuest[-1]  # get last element

		# Concaténation des deux dernières sortie
		# (histoire et question)
		out = th.cat((outStory, outQuest))
		out = self.lin(out)
		return self.act(out)
