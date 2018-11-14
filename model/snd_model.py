import torch as th
import torch.nn as nn
import torch.nn.functional as F


class SndModel(nn.Module):

	def __init__(self, vocab_size, embedding_size, story_hidden_size, quest_hidden_size, story_max_len, quest_max_len):
		super(SndModel, self).__init__()

		self.vocab_size = vocab_size
		self.embedding_size = embedding_size
		self.story_hidden_size = story_hidden_size
		self.quest_hidden_size = quest_hidden_size
		self.story_max_len = story_max_len
		self.quest_max_len = quest_max_len
		self.out_channel_conv = 20

		self.embedding_story = nn.Embedding(self.vocab_size, self.embedding_size)
		self.rnn_story_1 = nn.RNN(self.embedding_size, self.story_hidden_size)

		self.embedding_quest = nn.Embedding(self.vocab_size, self.embedding_size)
		self.rnn_quest = nn.RNN(self.embedding_size, self.quest_hidden_size)

		kernel_sizes = [3, 4, 5]
		self.conv_story = nn.ModuleList(
			[nn.Conv2d(1, self.out_channel_conv, (k, self.story_hidden_size)) for k in kernel_sizes])
		self.maxpool_story = nn.ModuleList(
			[nn.MaxPool1d(self.story_max_len + self.quest_max_len - k + 1, self.out_channel_conv) for k in kernel_sizes])

		self.lin = nn.Linear(self.out_channel_conv*len(kernel_sizes), self.vocab_size)
		self.act = nn.LogSoftmax(dim=0)

	def forward(self, inp):
		story, quest, _ = inp
		# story.shape=(nb_mot_story)
		# quest.shape=(nb_mot_quest)

		out_story = self.embedding_story(story)
		# outStory.shape = (nb_mot_story, taille_embedding)
		out_story, _ = self.rnn_story_1(out_story.unsqueeze(1), th.zeros(1, 1, self.story_hidden_size))
		# outStory.shape = (nb_mot, 1, story_hidden_size)

		out_quest = self.embedding_quest(quest)
		out_quest, _ = self.rnn_quest(out_quest.unsqueeze(1), th.zeros(1, 1, self.quest_hidden_size))

		out_story = out_story.squeeze(1) # outStory.shape = (nb_mot, story_hidden_size)
		out_quest = out_quest.squeeze(1)

		out_story = F.pad(out_story, (0, 0, 0, self.story_max_len - out_story.size(0)), 'constant', 0)
		out_quest = F.pad(out_quest, (0, 0, 0, self.quest_max_len - out_quest.size(0)), 'constant', 0)

		outs = th.cat((out_story, out_quest)).unsqueeze(0).unsqueeze(0)

		outs = [conv2d(outs).squeeze(3) for conv2d in self.conv_story]
		outs = [pool(out).squeeze(0).squeeze(1) for pool, out in zip(self.maxpool_story, outs)]

		out = th.cat(outs)
		out = self.lin(out)
		return self.act(out)


