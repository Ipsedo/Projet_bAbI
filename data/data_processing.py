import numpy as np

f= open("./res/tasks_1-20_v1-2/en/qa1_single-supporting-fact_train.txt","r")

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

