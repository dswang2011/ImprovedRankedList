import io
import numpy as np
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
