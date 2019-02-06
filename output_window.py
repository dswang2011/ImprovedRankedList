from py4j.java_gateway import JavaGateway
from utils import get_perturbed_phrases
import re
from word_embedding import *

# from functions import *
# gateway.launch_gateway(port=25333)
class IndriAPI(object):
    def __init__(self, index_dir_path):
        gateway = JavaGateway()

        indri_entry_point = gateway.entry_point
        indri_entry_point.setIndexDir(index_dir_path)
        self.indri = indri_entry_point.getIndriAPI()

    def get_co_occur_doc_count(self, dependency, window_size):
        counts = self.indri.getCoOccurDocCount(dependency,window_size)
        return counts

    def get_doc_id(self, doc_name):
        doc_id = self.indri.getDocID(doc_name)
        return(doc_id)

    def get_doc_name(self, doc_id):
        doc_name = self.indri.getDocNo(doc_id)
        return(doc_name)

    def get_doc_frequency(self, term):
        df = self.indri.getDocFrequency(term)
        return(df)

    def get_doc_length(self, doc_name):
        doc_length = self.indri.getDocLength(doc_name)
        return(doc_length)

    def get_term_count(self, doc_name, term):
        term_count = self.indri.getTermCount(doc_name, term)
        return(term_count)

    def get_document_array(self, doc_name):
        document_array = self.indri.getDocumentTermArray(doc_name)
        return(document_array)

    def get_co_occur_count_in_collection(self,dependency):
        count = self.indri.getCoOccurCount(dependency)
        return(count)

    def get_co_occur_count(self, terms, document_array, window_size = 5):
        dependency_string = ''.join(terms)
        count = self.indri.getCoOccurCountInDoc(document_array, window_size, dependency_string)
        return(count)

    def get_query_array(self, query):
        term_array = self.indri.getQueryTermArray(query)
        return(term_array)

    def get_indri_ranking(self, query, top_k = 200):
        doc_list = self.indri.generateTopKResults(query, top_k)
        return(doc_list)

    def get_tf_idf_4doc(self,doc_id):
        tf_idf = self.indri.getTFIDFRep4Doc(doc_id)
        return(tf_idf)

    def get_tf_idf_4docs(self,doc_id_list_str):
        tf_idf = self.indri.getTFIDFRep4Docs(doc_id_list_str)
        return(tf_idf)

    def get_collection_tf(self, term):
        tf = self.indri.getTermFrequency(term)
        return(tf)

    def get_collection_doc_count(self):
        doc_count = self.indri.getCollectionDocCount()
        return(doc_count)

    def get_collection_term_count(self):
        count = self.indri.getCollectionTermCount()
        return(count)

    def get_context_list(self,dependency, window_size = 5,max_contexts_num = 20):
        contexts = self.indri.getContexts(dependency,window_size,max_contexts_num)
        return(contexts)

def main():
    index_dir_path = "C:\\L\\data\\random_5"
    indri = IndriAPI(index_dir_path)
    print(indri.get_context_list('ivory tower',5))
    # print(indri.get_doc_frequency('day'))
    # print(indri.get_doc_frequency('dai'))
    # print(get_context_TFIDF(contexts, indri))
    # print(contexts)

    # print(len(docs))
import numpy as np
def get_vect4phrase(file_path):
	phrase4vect = {}
	with open(file_path,'r',encoding='utf8') as fr:
		phrase = ''
		vect_strs = []
		content = fr.readlines()
		for line in content:
			if line.strip()=='':
				continue
			strs = line.split('\t')
			if len(strs)>2 and strs[1].strip() in ['o','t']:
				if len(vect_strs)>0:
					vect_str = ' '.join(vect_strs).replace('[[','').replace(']]','').replace('  ',' ')
					phrase4vect[phrase] = np.array([np.fromstring(vect_str,dtype=float,sep=' ')])
				# give new values
				phrase = strs[0].strip()
				vect_strs.clear()
				vect_strs.append(strs[2].strip())
			else:
				vect_strs.append(line.strip())
	return phrase4vect


def get_phrase_list(file_path):
	phrase_list = []
	with open(file_path,'r',encoding='utf8') as fr:
		content = fr.readlines()
		for line in content:
			if line.strip()=='':
				continue
			phrase_list.append(line.strip())
	return phrase_list
	
def write_line(file_path,content):
	with open(file_path,'a',encoding='utf8') as fw:
		fw.write(content+'\n')
		fw.close()

def get_orig_window(phrase,window):
	strs = window.split(' ')
	strs2 = strs[:8]+[phrase]+strs[8:]
	return ' '.join(strs2)


from improved_rank_list import ImprovedRankList
if __name__ == '__main__':
	# indri
	index_dir_path = "C:\\L\\data\\random_5"
	indri = IndriAPI(index_dir_path)
	# rank list
	rank_list = ImprovedRankList()
	rank_list.parse_config('config/config.ini')
	rank_list.initialize()

	phrase2tfidf = get_phrase2vect_dict('p_tfidf.txt','tfidf')
	phrase2embed = get_phrase2vect_dict('p_word_embedding.txt','word_embedding')
	scenario2tfidf = get_scenario2vect_dict('s_tfidf.txt','tfidf')
	scenario2embed = get_scenario2vect_dict('s_word_embedding.txt','word_embedding')
	print('dics:',len(phrase2tfidf),len(phrase2embed),len(scenario2tfidf),len(scenario2embed))


	phrase4vect = get_vect4phrase('resources/vect_word2vec.txt')
	flag = 0
	phrase_set = {}
	# read phrase list; data struct: 1) phrase 2) scenario 3) avg_comp score 4) binary/triple comp score
	with open('input/avg_data.txt','r',encoding='utf8') as fr:
		content = fr.readlines()
		for line in content:
			if line.strip()=='':
				continue
			strs = line.split('\t')
			if len(strs)<2:
				continue
			p = strs[0].strip()
			s = strs[1].strip()

			phrase_windows = []
			# vect for p
			if p not in phrase2tfidf.keys() and p not in phrase_set.keys():
				
				# get matched windows and vectors for 
				phrase_windows = rank_list.get_window_list(p)

				p_tfidf = rank_list.get_context_rep_manual(phrase_windows,'tfidf')
				p_word_emb = rank_list.get_context_rep_manual(phrase_windows,'word_embedding')
				write_line('p_tfidf.txt',p+'\to\t'+str(p_tfidf))
				write_line('p_word_embedding.txt',p+'\to\t'+str(p_word_emb).replace('\n',' '))
				phrase_set[p] =p
			# perturbs
			perturbs = get_perturbed_phrases(p)
			print('now process:',p)
			pruned_perturbs = rank_list.prune_perturbed_phrase(perturbs)
			write_line('perturb_dict.txt',p+'\t'+str(pruned_perturbs).strip())
			for perturb in pruned_perturbs:
				if perturb not in phrase2tfidf.keys() and perturb not in phrase_set.keys():
					perturb_windows = rank_list.get_window_list(perturb)
					p_tfidf = rank_list.get_context_rep_manual(perturb_windows,'tfidf')
					p_word_emb = rank_list.get_context_rep_manual(perturb_windows,'word_embedding')
					write_line('p_tfidf.txt',perturb+'\tt\t'+str(p_tfidf))
					write_line('p_word_embedding.txt',perturb+'\tt\t'+str(p_word_emb).replace('\n',' '))
					phrase_set[perturb] = perturb

			# vect for s
			if len(p)+1>=len(s):
				continue
			if p+'\t'+s in scenario2embed.keys():
				continue
			if len(phrase_windows)==0:
				phrase_windows = rank_list.get_window_list(p)
			pattern = re.compile(p, re.IGNORECASE)
			s_strip = pattern.sub('', s)
			s_strip = s_strip.replace(' s ','')
			
			k_shrink = max(int(len(phrase_windows)/(2**len(s_strip.split(' ')))),10)
			s_windows = rank_list.get_matched_windows(s_strip, phrase_windows, n_match = k_shrink)
			mached_windows = []
			for i in range(len(s_windows)):
				mached_windows.append(s_windows[i][0])
			s_tfidf = rank_list.get_context_rep_manual(mached_windows,'tfidf')
			s_word_emb = rank_list.get_context_rep_manual(mached_windows,'word_embedding')
			write_line('s_tfidf.txt',p+'\t'+s+'\t'+str(s_tfidf))
			write_line('s_word_embedding.txt',p+'\t'+s+'\t'+str(s_word_emb).replace('\n',' '))



# gateway.entry_point.print()             # connect to the JVM
  # create a java.util.Random instance
# number1 = random.nextInt(10)              # call the Random.nextInt method
# number2 = random.nextInt(10)
# print(number1,number2)
# addition_app = gateway.entry_point        # get the AdditionApplication
# addition_app.addition(number1,number2)
