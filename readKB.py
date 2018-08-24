
file_prepared_KB = 'sampling_KB.txt'
test_data = 'small_test.txt' 

def get_candidate_pages(phrase, knowledge_base):
    if phrase.strip() in knowledge_base:
    	return knowledge_base[phrase]
    else:
    	return []

def read_test_data(test_data):
	phrases = []
	scenarios = []
	labels = []
	with open(test_data,'r') as f:
		for line in f:
			strs = line.split('\t')
			if len(strs)>2:		
				phrases.append(strs[0].strip())
				scenarios.append(strs[1].strip())
				labels.append(strs[2].strip())
	return phrases,scenarios,labels

def get_prepared_KB(file_prepared_KB):
	phrase2candidates = {}
	with open(file_prepared_KB,'r') as f:
		p = ''
		candi_list = []
		for line in f:
			if line.startswith('=='):
				# process last one
				if p != '':
					phrase2candidates[p] = candi_list
				# process next one
				strs = line.split('\t')
				p = strs[1].strip()
				candi_list.clear()	# clear the list
			else:
				strs = line.split('\t')
				if len(strs)>1: 
					explain = strs[1].replace("==ss==","")
					explain = explain.replace("==DB==","")
					candi_list.append(explain)
	return phrase2candidates


# test case:
knowledge_base = get_prepared_KB(file_prepared_KB)
phrases,scenarios,labels = read_test_data(test_data)
for i in range(len(phrases)):
	candi_list = get_candidate_pages(phrases[i],knowledge_base)
	print(phrases[i],":",labels[i])