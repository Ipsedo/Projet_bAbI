import numpy as np
import re

f= open("../res/tasks_1-20_v1-2/en/qa1_single-supporting-fact_train.txt","r")


def prepare_data(file):
	data_train = []
	cpt = 0
	tmp = []
	for line in file:
		tmp.append(line)
		cpt += 1
	return np.array(data_train)
	file.close()


data_train = prepare_data(f)
print(data_train)


def split(sent):
	sent = sent.lower()
	sent = re.sub(r'\d', '', sent)
	return [w.strip(" ") for w in re.split('(\W+)', sent) if w.strip(" ")]


def split_sentence(data):
	res = []
	for story, quest, ans in data:
		new_story = split(story)
		new_quest = split(quest)
		res.append((new_story, new_quest, ans))
	return res


def make_vocab_and_transform_data(splitted_data):
	vocab = {}
	res = []
	for story, quest, ans in splitted_data:
		new_story = []
		for w in story:
			if w not in vocab:
				vocab[w] = len(vocab)
			new_story.append(vocab[w])

		new_quest = []
		for w in quest:
			if w not in vocab:
				vocab[w] = len(vocab)
			new_quest.append(vocab[w])

		if ans not in vocab:
			vocab[ans] = len(vocab)
		new_ans = vocab[ans]

		res.append((new_story, new_quest, new_ans))
	return vocab, res



