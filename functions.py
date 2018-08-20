
# functions 

prase2context = {}


def get_matched_contexts(phrase_context, candidate_contexts_list, threshold):
    matched_contexts_dic = {}
    for candidate_context in candidate_contexts_list:
        score = ranked_list_similarity(phrase_context, candidate_context, 1)
        if score > threshold:
            matched_contexts_dic[candidate_context] = score

    return matched_contexts_dic



################## Qiuchi: ###################

# define of the context as a dict with terms and weights. 
class Context(Object):
    def __init__(self,phrase):
        self.phrase=phrase
        self.term2weight=term2weight{}


# 1. foir term weight, we comput TF_IDF with the corpus staistics (like TREC disks 4-5).

### 2. get context for a phrase from corpus ###
# input: a phrase or a single term   
# output: 5-window cumulative context from corpus. context can be <word,frequency> dict. 
def get_context(corpus, phrase):
    term_weight_dict = {}
    return term_weight_dict

 def get_combined_context4p(corpus, phrase):
    term_weight_dict = {}
    return term_weight_dict

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

## 4. get synomym, which is used by 3. (I have Java version of this.) 
# input: 
# output:
def get_synonym_terms(wordnet, term):
    term_list = []
    return(term_list)

## 5. similarity between two phrases. why do you use weight? I removed that.
# input:
# output:
 def similarity_rankList(phrase1, phrase2):
    score = 0
    return score

def similarity_wordvector(phrase1, phrase2):
    score = 0
    return score
