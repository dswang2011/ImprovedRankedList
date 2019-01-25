

def write_line(file_path, content):
	with open(file_path,'a',encoding='utf8') as fw:
		fw.write(content+'\n')
		fw.close()

phrase2sent = {}
with open('avg_data_orig.txt','r',encoding='utf8') as fr:
	content = fr.readlines()
	for line in content:
		if line.strip()=='':
			continue
		strs = line.split('\t')
		phrase2sent[strs[0].strip()] = line.strip()

with open('1042_orig.txt','r',encoding='utf8') as fr:
	content = fr.readlines()
	for line in content:
		if line.strip()=='':
			continue
		strs = line.split('\t')
		if strs[0].strip() in phrase2sent.keys():
			write_line('avg_data.txt',phrase2sent[strs[0].strip()])
		else:
			write_line('avg_data.txt',strs[0].strip()+'\t'+line.strip())


