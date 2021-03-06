import torch as th
import re


def use_cuda():
	"""

	:return: True if we need to use CUDA
	"""
	return False#th.cuda.is_available()


def split_file(file):
	"""

	:param file: A file object in reading mode
	:return: All the sentences in the file with \n removed
	"""
	res = []
	out = file.readlines()
	for s in out:
		res.append(s.replace("\n", ''))
	return res


def process_data(data):
	"""
	Nous traitons les phrases de la manière suivante :
	Une histoire de 15 phrases (10 affirmations et 5 questions)
	sera divisé en 5 sous histoires : la première ne contenant
	que deux affirmations et la question les suivante ainsi
	que la réponse. La deuxième les quatre premières affirmations,
	leur question et sa réponse etc. Jusqu'à la dernière sous histoire
	contenant toute les affirmations et la dernière question-réponse
	:param data: The list of sentence
	:return: A list of tuple(story, question, answer)
		where story and question are list of sentence
	"""
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
	"""
	split and process the sentence :
	- pass to lower all character
	- remove the id
	- split with non-word string
	:param sent: the sentence to split
	:return: list of word
	"""
	sent = sent.lower()
	sent = re.sub(r'\d', '', sent)
	return [w.strip(" ") for w in re.split('(\W+)', sent) if w.strip(" ")]


def split_sentence(data):
	"""
	Split all the sentence to list of word

	:param data: list of tuple(story, question, answer)
		where story is list of sentence, idem for question and answer an word
	:return: list of tuple(story, question, answer)
		story : list of word (string)
		question : list of word (string)
		answer : word
	"""
	res = []
	for story, quest, ans in data:
		new_story = split(story)
		new_quest = split(quest)
		res.append((new_story, new_quest, ans))
	return res


def make_vocab_and_transform_data(splitted_data):
	"""
	Pass the words to word-id and create at the same time the vocabulary

	:param splitted_data: list of tuple(story, question, answer)
		story : list of word (string)
		question : list of word (string)
		answer : word
	:return: list of tuple(tuple(story, question, answer), vocab)
		story : list of word-id (int)
		question : list of word-id (int)
		answer : word-id
		vocab : dict string -> int
	"""
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

		t_story = th.LongTensor(new_story)
		t_quest = th.LongTensor(new_quest)
		if use_cuda():
			t_story = t_story.cuda()
			t_quest = t_quest.cuda()
		res.append((t_story, t_quest, new_ans))
	return vocab, res


def make_data_with_vocab(splitted_data, vocab):
	"""
		Pass the words to word-id and create at the same time the vocabulary

		:param splitted_data: list of tuple(story, question, answer)
			story : list of word (string)
			question : list of word (string)
			answer : word
		:param vocab: dict string -> int
			give the word-id
		:return: list of tuple(story, question, answer)
			story : list of word-id (int)
			question : list of word-id (int)
			answer : word-id
		"""
	res = []
	for story, quest, ans in splitted_data:
		new_story = []
		for w in story:
			new_story.append(vocab[w])

		new_quest = []
		for w in quest:
			new_quest.append(vocab[w])

		new_ans = vocab[ans]

		t_story = th.LongTensor(new_story)
		t_quest = th.LongTensor(new_quest)
		if use_cuda():
			t_story = t_story.cuda()
			t_quest = t_quest.cuda()
		res.append((t_story, t_quest, new_ans))
	return res


def get_story_quest_max_len(prepared_data):
	"""
	Get story and question max len
	:param prepared_data: Processed data
	:return: max_story_len, max_question_len
	"""
	max_story = 0
	max_quest = 0
	for s, q, _ in prepared_data:
		max_story = len(s) if len(s) > max_story else max_story
		max_quest = len(q) if len(q) > max_quest else max_quest
	return max_story, max_quest