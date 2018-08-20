################## Qiuchi: ###################


# 1. foir term weight, we comput TF_IDF with the corpus staistics (like TREC disks 4-5).

### 2. get context for a phrase from corpus ###
# input: a phrase or a single term   
# output: 5-window cumulative context from corpus. weight should be added up for each word.
def get_context_TFIDF(corpus, phrase, index):
    term_weight_dict = {}
    return term_weight_dict

 def get_combined_WordEmbedding(corpus, phrase, table_dict):
    context_vector = []	# 300 dimention
    return context_vector

## 3. get perturbated phrase for phrase p. The question is we should investigate how to filter out meanless ones (or top-k, k=5/6?).
# input: phrase p
# output： a list of phrases
def get_perturbated_phrases(p):
	perturbed_phrases = []
    for term in terms:
    	synonym_terms = get_synonym_terms(wordnet, term) #get synonym terms from knowledge base

        for synonym_term in synonym_terms:
            perturbed_phrase = p.replace(term,synonym_term)
            perturbed_phrase_list.append(perturbed_phrase)
    return perturbed_phrase_list

## 4. get synomym, which is used by 3. Invoke WordNet api.
# input: 
# output:
def get_synonym_terms(wordnet, term):
    term_list = []
    return(term_list)


