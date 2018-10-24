import numpy as np
import re

f= open("../res/tasks_1-20_v1-2/en/qa1_single-supporting-fact_train.txt","r")


def split_file(file):
	res = []
	out = file.readlines()
	for s in out:
		res.append(s.replace("\n", ''))
	return res

def process_data(data):
	res = []
	total = len(data)
	i = 0
	while total != i:
		max = [3,6,9,12,15]
		for m in max:
			story = ""
			quest = ""
			for j in range(m):
				if (j+1)%3 != 0:
					s = data[i+j]
					story += s
				else:
					line_of_q = data[i+j]
					line_of_q = line_of_q.split("\t")
					quest = line_of_q[0]
					ans = line_of_q[1]
			res.append((story, quest, ans))
		i += 15
	return res

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

data_train = process_data(split_file(f))

print(data_train[10])

data=split_sentence(data_train)
print(data[2])

vocab, prepared_data = make_vocab_and_transform_data(data)

print(prepared_data[10])

