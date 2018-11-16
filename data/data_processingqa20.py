import torch as th
import re

"""
Identique à la fonction "split_file" du fichier "data_processing.py"
"""
def split_file(file):
	res = []
	out = file.readlines()
	for s in out:
		res.append(s.replace("\n", ''))
	return res
"""
Fonction permettant de split les données au niveau des espaces.
"""
def split_space(data):
	res = []
	for s in data:
		res.append(s.split(" "))
	return res
"""
Création de liste permettant le bon déroulement du split 
(variable) pour chacunes des histoires du fichier.
"""
def mk_utils(data):
	l1 = []
	l2 = []
	res = []
	resF = [0]
	res.append(0)
	for i in range(len(data)):
		l1.append(data[i][0])
	for j in range(len(data) - 1):
		l2.append(data[j+1][0])
	for i in range(len(l2)):
		if int(l1[i]) > int(l2[i]):
			res.append(int(l1[i]))
		if i == len(l2)-1:
			res.append(int(l2[i]))
	for i in range(len(res)-1):
		resF.append(res[i+1] + resF[i])
	del resF[-1]
	del res[0]
	return resF, res
"""
Cette fonction est basée sur le même fonctionnement que 'process data' 
dans 'data_processing'. Le but étant de constituer des triplet de type
([1,2,3,4,5,6], [2,8,5,6,5], 15) où chaque numéro représente l'identifier du mot
du vocabulaire. La première liste pour les phrases d'informations.
La deuxième liste pour représenter la question.
Le dernier id pour la réponse associée.
La taille des histoires étant variable ici on traite des tailles de contexte qui varient
entre 20 et 24 lignes.
"""
def process_data(data):
	data_tier = split_space(data)
	l_batch1, l_batch2 = mk_utils(data_tier)
	res = []
	for i, j in zip(l_batch1, l_batch2):
		max = []
		for k in range(2, j+2, 2):
			max.append(k)
		for l in max:
			story = ""
			quest = ""
			ans = ""
			for m in range(l):
				if(m+1)%2 != 0:
					s = data[i+m]
					story += s
				else:
					line_of_q = data[i+m]
					line_of_q = line_of_q.split("\t")
					quest = line_of_q[0]
					ans = line_of_q[1]
			res.append((story, quest, ans))
	return res

"""
Identique à la fonction "split" du fichier "data_processing.py"
"""
def split(sent):
	sent = sent.lower()
	sent = re.sub(r'\d', '', sent)
	return [w.strip(" ") for w in re.split('(\W+)', sent) if w.strip(" ")]

"""
Identique à la fonction "split_sentence" du fichier "data_processing.py"
"""
def split_sentence(data):
	res = []
	for story, quest, ans in data:
		new_story = split(story)
		new_quest = split(quest)
		res.append((new_story, new_quest, ans))
	return res

"""
Identique à la fonction "make_vocab_and_transform_data" du fichier "data_processing.py"
"""
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

		res.append((th.LongTensor(new_story), th.LongTensor(new_quest), new_ans))
	return vocab, res

"""
Identique à la fonction "make_data_with_vocab" du fichier "data_processing.py"
"""
def make_data_with_vocab(splitted_data, vocab):
	res = []
	for story, quest, ans in splitted_data:
		new_story = []
		for w in story:
			new_story.append(vocab[w])

		new_quest = []
		for w in quest:
			new_quest.append(vocab[w])

		new_ans = vocab[ans]

		res.append((th.LongTensor(new_story), th.LongTensor(new_quest), new_ans))
	return res