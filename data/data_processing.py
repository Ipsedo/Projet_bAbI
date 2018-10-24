import numpy as np

f= open("./res/tasks_1-20_v1-2/en/qa1_single-supporting-fact_train.txt","r")

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

data_train = process_data(split_file(f))

print(data_train[0])

