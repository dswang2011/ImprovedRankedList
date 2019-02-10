import io
import numpy as np
import ast
def form_matrix(file_name):
    word_list = []
    ll = []
    with io.open(file_name, 'r',encoding='utf-8') as f:
        for line in f:
            word, vec = line.split(' ', 1)
            ll.append(np.fromstring(vec, sep=' '))
            word_list.append(word)
        matrix = np.asarray(ll)
    return matrix, word_list


def get_phrase2vect_dict(phrase2idf_file, context_type):
	phrase2vect = {}
	with open(phrase2idf_file,'r',encoding='utf8') as fr:
		for line in fr:
			if line.strip()=='':
				continue
			strs = line.strip().split('\t')
			if context_type=='tfidf':
				phrase2vect[strs[0].strip()] = ast.literal_eval(strs[2].strip())
			elif context_type=='word_embedding':
				vect = np.fromstring(strs[2].strip().replace('[[','').replace(']]',''),sep=' ')
				phrase2vect[strs[0].strip()] = vect
	return phrase2vect

def get_scenario2vect_dict(phrase2idf_file, context_type):
	scenario2vect = {}
	with open(phrase2idf_file,'r',encoding='utf8') as fr:
		for line in fr:
			if line.strip()=='':
				continue
			strs = line.strip().split('\t')
			if context_type=='tfidf':
				scenario2vect[strs[0].strip()+'\t'+strs[1].strip()] = ast.literal_eval(strs[2].strip())
			elif context_type=='word_embedding':
				vect = np.fromstring(strs[2].strip().replace('[[','').replace(']]',''),sep=' ')
				scenario2vect[strs[0].strip()+'\t'+strs[1].strip()] = vect
	return scenario2vect


p2vect = get_phrase2vect_dict('resources/p_tfidf.txt',context_type='tfidf')
print(len(p2vect))
# p2vect = get_phrase2vect_dict('resources/p_word_embedding.txt',context_type='word_embedding')
# print(len(p2vect))
# p2vect = get_scenario2vect_dict('resources/s_tfidf.txt',context_type='tfidf')
# print(len(p2vect))
# p2vect = get_scenario2vect_dict('resources/s_word_embedding.txt',context_type='word_embedding')
# print(len(p2vect))

# def get_wordvec(path_to_vec, word2id=None):
#     matrix, word_list = form_matrix(path_to_vec)
#     coefficients_matrix = np.transpose(matrix)
#     word_vec = {}
#      # if word2vec or fasttext file : skip first line "next(f)"
#     if word2id == None:
#         print('program goes here!')
#         for word in word_list:
#             word_vec[word] = coefficients_matrix[:, word_list.index(word)]
#     else:
#         for word in word_list:
#             if word in word2id:
#                    word_vec[word] = coefficients_matrix[:, word_list.index(word)]

#     logging.info('Found {0} words with word vectors, out of \
#         {1} words'.format(len(word_vec), len(word2id)))
#     return word_vec

# def get_word_vector(term,matrix,word_list):
#     if term in word_list:
#         word_vec = matrix[word_list.index(term),:]
#         return(word_vec)
#     else:
#         return None

# path_to_vec = 'glove/glove.6B.50d.txt'
# matrix, word_list = form_matrix(path_to_vec)
# word_vec = get_word_vector('today', matrix, word_list)
# print(word_vec.shape)
