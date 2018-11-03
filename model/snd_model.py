import torch as th
import torch.nn as nn


class SndModel(nn.Module):

	def __init__(self, vocab_size, embedding_size, story_hidden_size, quest_hidden_size):
		super(SndModel, self).__init__()

		self.vocab_size = vocab_size
		self.embedding_size = embedding_size
		self.story_hidden_size = story_hidden_size
		self.quest_hidden_size = quest_hidden_size

		self.embedding_story = nn.Embedding(self.vocab_size, self.embedding_size)
		self.rnn_story_1 = nn.RNN(self.embedding_size, self.story_hidden_size)
		self.rnn_story_2 = nn.RNN(self.embedding_size, self.story_hidden_size)

		self.embedding_quest = nn.Embedding(self.vocab_size, self.embedding_size)
		self.rnn_quest = nn.RNN(self.embedding_size, self.quest_hidden_size)

		self.lin = nn.Linear(2*self.story_hidden_size + self.quest_hidden_size, self.vocab_size)
		self.act = nn.LogSoftmax(dim=0)

	def forward(self, inp):
		story, quest, _ = inp
		# story.shape=(nb_mot_story)
		# quest.shape=(nb_mot_quest)

		outStory = self.embedding_story(story)
		# outStory.shape = (nb_mot_story, taille_embedding)
		outStory1, _ = self.rnn_story_1(outStory.unsqueeze(1), th.zeros(1, 1, self.story_hidden_size))
		outStory2, _ = self.rnn_story_2(reversed(outStory).unsqueeze(1), th.zeros(1, 1, self.story_hidden_size))
		# outStory.shape = (nb_mot, 1, story_hidden_size)

		outQuest = self.embedding_quest(quest)
		outQuest, _ = self.rnn_quest(outQuest.unsqueeze(1), th.zeros(1, 1, self.quest_hidden_size))

		outStory1 = outStory1.squeeze(1) # outStory.shape = (nb_mot, story_hidden_size)
		outStory2 = outStory2.squeeze(1)
		outQuest = outQuest.squeeze(1)

		outStory1 = outStory1[-1]  # get last element
		outStory2 = outStory2[0]
		outQuest = outQuest[-1]  # get last element

		out = th.cat((outStory1, outStory2, outQuest))
		out = self.lin(out)
		return self.act(out)


