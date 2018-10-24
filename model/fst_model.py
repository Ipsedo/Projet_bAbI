import torch as th
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, vocab_size, embedding_size, story_hidden_size, quest_hidden_size):
        super(MyModel, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.story_hidden_size = story_hidden_size
        self.quest_hidden_size = quest_hidden_size

        self.embedding_story = nn.Embedding(self.vocab_size, self.embedding_size)
        self.rnn_story = nn.RNN(self.embedding_size, self.story_hidden_size)

        self.embedding_quest = nn.Embedding(self.vocab_size, self.embedding_size)
        self.rnn_quest = nn.RNN(self.embedding_size, self.quest_hidden_size)

        self.lin = nn.Linear(self.story_hidden_size + self.quest_hidden_size, self.vocab_size)
        self.act = nn.Softmax(dim=0)

    def forward(self, input):
        """
        input = tuple(story, quest, reponse)
        story = sac d'indice de mot
            Pour la premiere question, on prend les 2 1eres affirmation
            Pour la 2e : on prend les 4 1eres affirmation
            etc.
        quest = sac de mot
        :param input:
        :return:
        """
        story, quest, _ = input
        # story.shape=(nb_mot_story)
        # quest.shape=(nb_mot_quest)

        outStory = self.embedding_story(story)
        # outStory.shape = (nb_mot_story, taille_embedding)
        outStory, hidden = self.rnn_story(outStory.unsqueeze(1), th.FloatTensor(1, 1, self.story_hidden_size))
        # outStory.shape = (nb_mot, 1, story_hidden_size)

        outQuest = self.embedding_quest(quest)
        outQuest, hidden = self.rnn_quest(outQuest.unsqueeze(1), th.FloatTensor(1, 1, self.quest_hidden_size))

        outStory = outStory.squeeze(1)
        # outStory.shape = (nb_mot, story_hidden_size)
        outQuest = outQuest.squeeze(1)

        outStory = outStory[-1] # get last element
        outQuest = outQuest[-1] # get last element

        out = th.cat((outQuest, outStory))
        out = self.lin(out)
        return self.act(out)








