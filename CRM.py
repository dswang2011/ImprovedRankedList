import os
import io,re,codecs
import numpy as np
import configparser
import argparse
from corpus_index import IndriAPI
from word_embedding import *
from knowledge_base import get_prepared_KB
from utils import *
import math
import re


class ImprovedRankList(object):

    def __init__(self):
        pass
    def parse_config(self, config_file_path):
        config = configparser.ConfigParser()
        config.read(config_file_path)
        config_common = config['COMMON']
        is_numberic = re.compile(r'^[-+]?[0-9.]+$')
        for key,value in config_common.items():
            result = is_numberic.match(value)
            if result:
                if type(eval(value)) == int:
                    value= int(value)
                else:
                    value= float(value)

            self.__dict__.__setitem__(key,value)

    def get_parameter_list(self):
        info=[]
        for k, v in self.__dict__.items():
            info.append("%s -> %s"%(k,str(v)))
        return info

    def initialize(self):
        self.knowledge_base = get_prepared_KB(self.knowledge_base_path,stemmed = self.stem_words)
        # print(self.knowledge_base)
        self.corpus_index = IndriAPI(self.index_dir_path)
        self.word_embedding = None
        if 'path_to_vec' in self.__dict__:
            self.word_embedding = form_matrix(self.path_to_vec)
        # get pre-stored vectors
        if self.context_type=='tfidf':
            self.p2v = get_phrase2vect_dict(self.phrase2idf_file,self.context_type)
            self.s2v = get_scenario2vect_dict(self.scenario2idf_file,self.context_type)
        elif self.context_type=='word_embedding':
            self.p2v = get_phrase2vect_dict(self.phrase2embed_file,self.context_type)
            self.s2v = get_scenario2vect_dict(self.scenario2embed_file,self.context_type)
        self.prepared_phrase2perturb = get_prepared_p2perturb(self.phrase2perturb_file)

    '''
    The main function
    '''
    def run(self):
        # Load Data from file
        # phrases,scenarios,labels = read_test_data(self.input_path)
        data = read_test_data(self.input_path)

        output_file = 'output.txt'
        if 'output_file' in self.__dict__:
            output_file = self.output_file

        file_writer = codecs.open(output_file,'w',encoding='utf8')
        scores = []
        targets = []
        # for phrase,scenario,label in zip(phrases,scenarios,labels):
        for item in data:
        	phrase=item[0]
        	scenario=item[1]
        	label = item[2]
        	#Compute the compositional score
        	score = self.rank_list_comp(phrase, scenario)
        	score = float(score)
        	scores.append(score)
        	targets.append(1.0-float(label))
        file_writer.write('{}\t{}\t{}\t{}\n'.format(phrase,scenario,score,label))
        #Compute the Pearson Correlation Coefficient between the outputs and the ground truth
        pearson_correlation_coefficient = pearson_correlation(scores,targets)
        print('corr:',pearson_correlation_coefficient)
        writer = codecs.open(self.pearson_correlation_file,'w')
        writer.write(str(pearson_correlation_coefficient))
        # print(pearson_correlation_coefficient)

    '''
    Compute the Compositional Score of a phrase
    Given the Phrase phrase and its Scenario
    '''
    def rank_list_comp(self, phrase, scenario):
        
        p_stem = phrase
        if self.stem_words in ['True','true']:
            p_stem = stem_words(phrase)
            scenario = stem_words(scenario)

        ######## Step1: Compute localized contextual representation ######
        # Get all windows containing phrase in the corpus

        # phrase_windows = self.get_window_list(p_stem)

        #The number of shrinked K for scenario
        # k_shrink = max(int(len(phrase_windows)/(2**len(scenario.split(' ')))),8)
        
        
        print('Computing the context representation.')
        # Identify the contexts related to the scenario
        # The output is a dict of {matched_context, sim_score} pairs
        # matched_context2sim_dic = self.get_matched_contexts(scenario, phrase_windows, n_match = k_shrink, threshold = 0)
        
        # (1) Compute the original scenario representation (consider as one window)
        scenario_rep = self.get_context_rep([scenario])


        # Compute the updated phrase context representation
        # phrase_context_rep = self.compute_updated_context(scenario_context_rep, matched_context2sim_dic, average = True, weight = self.scenario_corpus_combine_weight)
#        phrase_context_rep = self.get_localized_contexts(scenario, phrase_windows, len(scenario))
       

        print(p_stem)
        # (2) phrase representation
        # phrase_norm_rep = self.p2v[p_stem]
        phrase_norm_rep = self.get_context_rep([p_stem])

        # (1) comput the normalized scenario context
        if p_stem+'\t'+scenario in self.s2v.keys():
        	scenario_norm_rep = self.s2v[p_stem+'\t'+scenario]
        else:
        	scenario_norm_rep = phrase_norm_rep

        # (3) combine  (set as 0, i.e., no scenario vect considered)
        phrase_context_rep = self.combine_context(scenario_norm_rep,phrase_norm_rep, self.scenario_corpus_combine_weight)

        ######## Step2: Adjusting localized context rep with KB ######
        # print(phrase_context_rep)
        if self.adapt_with_knowledge_base in ['True','TRUE']:
            print('Recomputing the context representation with Knowledge base.')

            # Get phrase diambiguation pages in the knowledge base
            candidate_page_list = self.get_candidate_pages(phrase)

            # Parallel model
            if self.adapt_pattern == 'parallel':
                # Find all (matched contexts, matching scores) above a given threshold
                matched_context2sim_dic = self.get_matched_contexts(scenario, candidate_page_list)

                # Generate the contexts of the original phrase based on matched candidate pages (do not need this one)
                phrase_kb_context_rep = self.compute_updated_context(scenario_rep, matched_context2sim_dic, average = False, weight = self.scenario_kb_combine_weight)

                # Combine the phrase context representation with its knowledge base representation
                phrase_context_rep = self.combine_context(phrase_context_rep, phrase_kb_context_rep,ratio = self.corpus_kb_combine_weight)

            # Sequential model
            elif self.adapt_pattern == 'sequential':
                # Find all (matched contexts, matching scores) above a given threshold
                matched_context2sim_dic = self.get_matched_contexts(scenario_norm_rep, candidate_page_list, n_match = 0, threshold = self.kb_matching_threshold)

                # Generate the contexts of the original phrase based on matched candidate pages
                phrase_context_rep = self.compute_updated_context(phrase_context_rep, matched_context2sim_dic, average = False, weight = self.phrase_kb_combine_weight)

        # Combine phrase representation with its context rep
        # phrase_context_rep = self.combine_context(phrase_rep, phrase_context_rep,ratio = self.phrase_context_ratio)
        
        ######## 3. get perturbed phrases ######
        # Generate the list of perturbed phrases.
        print('Get perturbed phrase list.')
        # perturbed_phrase_list = get_perturbed_phrases(phrase,self.stem_words)
        perturbed_phrase_list = self.prepared_phrase2perturb[p_stem]
        print('perturb size:',len(perturbed_phrase_list))

        print('Compute output score for determining CD/NCD.')
        output_score = 0

        # Traverse the perturbed phrases
        for perturbed_phrase in perturbed_phrase_list:
            if self.stem_words in ['True','true']:
                perturbed_phrase = stem_words(perturbed_phrase)
            # perturb_rep = self.get_context_rep([perturbed_phrase])
            # pertub_window_list = self.get_window_list(perturbed_phrase)
            # perturb_context_rep = self.get_context_rep(pertub_window_list)
            # perturb_context_rep = self.combine_context(perturb_rep,perturb_context_rep,ratio = self.phrase_context_ratio)
            # perturb_context_rep = self.p2v[perturbed_phrase]
            perturb_context_rep = self.get_context_rep([perturbed_phrase])
            # 4. calculate sim between phrase and perturb (sum and avg)
            output_score = output_score + self.get_context_similarity(phrase_context_rep, perturb_context_rep)
#            print(self.get_context_similarity(phrase_context_rep, context_rep))
#            print('6')
        print('output_score = {}'.format(output_score))
        avg_perturb_score = 0 # default CD score
        if len(perturbed_phrase_list)>0:
            avg_perturb_score = output_score/len(perturbed_phrase_list)
        if avg_perturb_score ==0:
            print('== [perturb_score is 0, suspicious phrase]==:',phrase,perturbed_phrase_list)
        print('avg_score = {}'.format(avg_perturb_score))
        return avg_perturb_score

    # Get window list for a phrase (or a perturbed phrase)
    def get_window_list(self, phrase):
        window_size = 15
        if 'window_size' in self.__dict__:
            window_size = self.window_size
        context_list = []
        if self.word_level:
            # Get the list of context words that are close to each word in the phrase
            words = phrase.split(' ')
            for word in words:
                word_str = word.strip()
                if len(word_str)>1:
                    temp_list = self.corpus_index.get_context_list(word_str, window_size = window_size)
                    context_list.extend(temp_list)
        else:
            # Get the list of context words that are close to the whole phrase
            context_list = self.corpus_index.get_context_list(phrase, window_size = window_size)
        return context_list

    # Get the representation of a list of contexts
    # Either as a tf-idf vector or as a word vector
    def get_context_rep(self, window_list):
        context_rep = None
        if self.context_type == 'tfidf':
            # Build the TF-IDF representation
            collection_doc_count = self.corpus_index.get_collection_doc_count()
            context_rep = {}
            for window in window_list:
                # print('window:',window)
                term_list = window.split()
                for term in term_list:
                    idf = math.log10(collection_doc_count/(self.corpus_index.get_doc_frequency(term)+1))
                    if term in context_rep:
                        context_rep[term] = context_rep[term]+idf
                    else:
                        context_rep[term] = idf

            topK_terms = 100   # default 1000
            if 'topK_terms' in self.__dict__:
                topK_terms = self.topK_terms

            # If the context representation has less than topK terms, then rank all the context terms
            topK_terms = min(topK_terms, len(context_rep))

            # Rank the context terms
            sorted_context_rep = sorted(context_rep.items(), key = lambda item:item[1], reverse = True)

            # Get the top k terms
            context_rep ={}
            L= sorted_context_rep[:topK_terms]
            for l in L:
                context_rep[l[0]] = l[1]

            # Divided by the number of contexts
            for term in context_rep:
                context_rep[term] = context_rep[term]/len(window_list)

        elif self.context_type == 'word_embedding':
            # Build the word vector representation
            matrix, word_list = self.word_embedding
            context_rep = np.zeros(shape = (1,matrix.shape[1]))
            if len(window_list) ==0:
                print('warning -- none of the words appear in the dictionary, returning zero vector.')
                return context_rep
            for context in window_list:
            	term_list = context.split()
            	index_list = []
            	for term in term_list:
            		if term in word_list:
            			index_list.append(word_list.index(term))
            	if len(index_list) == 0:
            		context_mean = np.zeros(shape = (1,matrix.shape[1]))
            	else:
            		context_mean = np.mean(matrix[index_list,:],axis = 0)
            	context_rep = context_rep + context_mean
           
            context_rep = context_rep/len(window_list)  
        return context_rep

    def get_context_rep_manual(self, window_list,context_type):
            context_rep = None
            collection_doc_count = self.corpus_index.get_collection_doc_count()    
            if context_type == 'tfidf':
                # Build the TF-IDF representation            
                context_rep = {}
                for window in window_list:
                    # print('window:',window)
                    term_list = window.split()
                    for term in term_list:
                        idf = math.log10(collection_doc_count/(self.corpus_index.get_doc_frequency(term)+1))
                        if term in context_rep:
                            context_rep[term] = context_rep[term]+idf
                        else:
                            context_rep[term] = idf

                topK_terms = 100   # default 1000
                if 'topK_terms' in self.__dict__:
                    topK_terms = self.topK_terms

                # If the context representation has less than topK terms, then rank all the context terms
                topK_terms = min(topK_terms, len(context_rep))

                # Rank the context terms
                sorted_context_rep = sorted(context_rep.items(), key = lambda item:item[1], reverse = True)

                # Get the top k terms
                context_rep ={}
                L= sorted_context_rep[:topK_terms]
                for l in L:
                    context_rep[l[0]] = l[1]

                # Divided by the number of contexts
                for term in context_rep:
                    context_rep[term] = context_rep[term]/len(window_list)

            elif context_type == 'word_embedding':
                # Build the word vector representation
                matrix, word_list = self.word_embedding
                context_rep = np.zeros(shape = (1,matrix.shape[1]))
                if len(window_list) ==0:
                    print('warning -- none of the words appear in the dictionary, returning zero vector.')
                    return context_rep
                for window in window_list:
                    term_list = window.split()
                    index_list = []
                    tot_idf = 0.0
                    for term in term_list:
                        if term in word_list:
                        	idf = math.log10(collection_doc_count/(self.corpus_index.get_doc_frequency(term)+1))
                        	index_list.append(word_list.index(term)*idf)
                        	tot_idf+=idf
                    if len(index_list) == 0:
                        context_mean = np.zeros(shape = (1,matrix.shape[1]))
                    else:
                        context_mean = np.sum(matrix[index_list,:],axis = 0)/tot_idf
                    context_rep = context_rep + context_mean  
                context_rep = context_rep/len(window_list)  
            return context_rep

#    # 300 dimension vector
#    def get_vect_rep(self, window_list):
#        context_rep = None
#        matrix, word_list = self.word_embedding
#        context_rep = np.zeros(shape = (1,matrix.shape[1]))
#        for context in window_list:
#            term_list = context.split()
#            # term_vector
#            index_list = []
#            for term in term_list:
#                if term in word_list:
#                    index_list.append(word_list.index(term))
#            context_mean = np.mean(matrix[index_list,:],axis = 0)
#            context_rep = context_rep + context_mean
#        # print(context_vector.shape)
#        context_rep = context_rep/len(window_list)   # 300 dimension
#        return context_rep

    def prune_perturbed_phrase(self, perturbed_phrases):
        topK_perturbed = 10
        if 'perturbation_num' in self.__dict__:
            topK_perturbed = self.perturbation_num

        topK_perturbed = min(topK_perturbed,len(perturbed_phrases))

        perturbed_phrases.sort(key=lambda x:self.corpus_index.get_co_occur_count_in_collection(x), reverse = True)
        # for phrase in perturbed_phrases:

        #     tf = self.corpus_index.get_co_occur_count_in_collection(phrase)
        return perturbed_phrases[0:topK_perturbed]

    def combine_context(self, context_rep_1,context_rep_2, ratio = 0.5):
        if self.context_type == 'tfidf':
            output = {}
            for term in context_rep_1:
                weight_1 = context_rep_1[term]
                weight_2 = 0
                if term in context_rep_2:
                    weight_2 = context_rep_2[term]
                
                weight = weight_1 * ratio + weight_2 * (1- ratio)
                output[term] = weight

        elif self.context_type == 'word_embedding':
            output = context_rep_1 *ratio + context_rep_2* (1-ratio)

        return output



    def get_context_similarity(self, context_rep_1,context_rep_2):
        similarity_score = 0
        if self.context_type == 'tfidf':
            sum_square_weight_1 = 0
            sum_square_weight_2 = 0
            inner_product = 0
            for term in context_rep_1:
                sum_square_weight_1 =  sum_square_weight_1+ context_rep_1[term]**2
                if term in context_rep_2:
                    inner_product = inner_product+ context_rep_2[term]* context_rep_1[term]
                    # print(context_rep_2[term],context_rep_1[term],inner_product)

            for term in context_rep_2:
                sum_square_weight_2 =  sum_square_weight_2+ context_rep_2[term]**2
            # print(inner_product)
            similarity_score = inner_product/(np.sqrt(sum_square_weight_1*sum_square_weight_2)+0.0001)
        
        elif self.context_type == 'word_embedding':
            similarity_score = np.inner(context_rep_1, context_rep_2)/(np.linalg.norm(context_rep_1)*np.linalg.norm(context_rep_2)+0.0001)
            similarity_score =  similarity_score[0][0]
        # print('context similarity score = {}'.format(similarity_score))
        return similarity_score

    def get_vect_similarity(self, vect_rep_1,vect_rep_2):
        similarity_score = 0
        if np.linalg.norm(vect_rep_1)*np.linalg.norm(vect_rep_2) == 0:
            return similarity_score
        else:
            similarity_score = np.inner(vect_rep_1, vect_rep_2)/(np.linalg.norm(vect_rep_1)*np.linalg.norm(vect_rep_2))
        # print(similarity_score[0][0])
        return similarity_score

    def get_matched_contexts(self, phrase_scenario, candidate_window_list, n_match = 0, threshold = 0):
        
        # If the input is already a context representation
        if type(phrase_scenario) == dict or type(phrase_scenario) == np.ndarray:
            phrase_context_rep = phrase_scenario
        else:
            phrase_context_rep = self.get_context_rep([phrase_scenario])
#        threshold = 0
#        if 'kb_matching_threshold' in self.__dict__:
#            threshold = self.kb_matching_threshold
        matched_contexts_pairs = []
        
        for candidate_window in candidate_window_list:
            candidate_context_rep = self.get_context_rep([candidate_window])
            score = self.get_context_similarity(phrase_context_rep, candidate_context_rep)
#            candidate_vect_rep = self.get_vect_rep([candidate_window])
#            score = self.get_vect_similarity(phrase_vect_rep, candidate_vect_rep)
            if score > threshold:
                matched_contexts_pairs.append((candidate_context_rep,score))
                # matched_contexts_dic[candidate_window] = score
#                total_score = total_score + score
                
        # Take the top n_match results, if there are more matched results
        if n_match>0 and n_match<len(matched_contexts_pairs):
            matched_contexts_pairs.sort(reverse = True, key = lambda x:x[1])
            matched_contexts_pairs = matched_contexts_pairs[:n_match]
            
        # Normalize scores
        total_score = 0
        for (result, score) in matched_contexts_pairs:
            total_score = total_score + score
         
        for i in range(len(matched_contexts_pairs)):
            matched_contexts_pairs[i] = (matched_contexts_pairs[i][0], matched_contexts_pairs[i][1]/total_score)
            
        return matched_contexts_pairs

    def get_matched_windows(self, phrase_scenario, candidate_window_list, n_match = 0, threshold = 0):
            
            # If the input is already a context representation
            if type(phrase_scenario) == dict or type(phrase_scenario) == np.ndarray:
                phrase_context_rep = phrase_scenario
            else:
                phrase_context_rep = self.get_context_rep([phrase_scenario])
    #        threshold = 0
    #        if 'kb_matching_threshold' in self.__dict__:
    #            threshold = self.kb_matching_threshold
            matched_window_pairs = []
            
            for candidate_window in candidate_window_list:
                candidate_context_rep = self.get_context_rep([candidate_window])
                score = self.get_context_similarity(phrase_context_rep, candidate_context_rep)
    #            candidate_vect_rep = self.get_vect_rep([candidate_window])
    #            score = self.get_vect_similarity(phrase_vect_rep, candidate_vect_rep)
                if score > threshold:
                    matched_window_pairs.append((candidate_window,score))
                    # matched_contexts_dic[candidate_window] = score
    #                total_score = total_score + score
                    
            # Take the top n_match results, if there are more matched results
            matched_window_pairs.sort(reverse = True, key = lambda x:x[1])
            if n_match>0 and n_match<len(matched_window_pairs):
                matched_window_pairs = matched_window_pairs[:n_match]
                
            # Normalize scores
            total_score = 0
            for (result, score) in matched_window_pairs:
                total_score = total_score + score
             
            for i in range(len(matched_window_pairs)):
                matched_window_pairs[i] = (matched_window_pairs[i][0], matched_window_pairs[i][1]/total_score)
                
            return matched_window_pairs

    # def get_localized_contexts(self, phrase_senario, all_window_list, scenario_length):
    #     threshold = 0
    #     if 'kb_matching_threshold' in self.__dict__:
    #         threshold = self.kb_matching_threshold
        
    #     # get K windows
    #     k_windows = []
    #     if scenario_length>0:
    #         # ranking first
    #         window_sim_pairs = []
    #         phrase_vect_rep = self.get_vect_rep([phrase_senario])
    #         for candidate_window in all_window_list:
    #             candidate_vect_rep = self.get_vect_rep([candidate_window])
    #             score = self.get_vect_similarity(phrase_vect_rep, candidate_vect_rep)
    #             if score > threshold:
    #                 window_sim_pairs.append((candidate_window,score))
    #         sorted_by_second = sorted(window_sim_pairs, key=lambda tup: tup[1])
    #         # get top 10
    #         for i in range(1,11):
    #             k_windows.append(sorted_by_second[len(sorted_by_second)-i][0])
    #     else:
    #         k_windows = all_window_list
    #     localized_context_rep = self.get_context_rep(k_windows)
    #     return localized_context_rep

    def compute_updated_context(self,phrase_context,matched_contexts_pairs, average = False, weight = 0.1):
#        weight = 0.1
#        if 'kb_combine_weight' in self.__dict__:
#            weight = self.kb_combine_weight
        if len(matched_contexts_pairs) == 0:
            return phrase_context
        
        avg_score = 1/len(matched_contexts_pairs)
        output_context = phrase_context.copy()
        if self.context_type == 'tfidf':
            for term in output_context:
                output_context[term] = output_context[term]* weight
        elif self.context_type == 'word_embedding':
            output_context = output_context * weight

        for context_rep,score in matched_contexts_pairs:
            if average:
                score = avg_score
            if self.context_type == 'tfidf':
                for term in context_rep:
                    if term in output_context:
                        output_context[term] = output_context[term] + context_rep[term]*score* (1-weight)
                    else:
                        output_context[term] = context_rep[term]*score* (1-weight)
            elif self.context_type == 'word_embedding':
                output_context = output_context + (1-weight)*score*context_rep

        return output_context

    # refined 
    def get_candidate_pages(self,phrase):
        window_size = 15
        if 'window_size' in self.__dict__:
            window_size = self.window_size
        res_pages = []
        if phrase.strip() in self.knowledge_base:
            pages = self.knowledge_base[phrase]
            for page in pages:
                # remove the repeating phrase (case insenstive)
                # page = page.replace(phrase,'')
                pattern = re.compile(phrase, re.IGNORECASE)
                page = pattern.sub('', page)
                page = page.replace(' s ','')

                # get only top window * 2 words
                words = page.split(' ')
                if len(words)<window_size*2:
                    res_pages.append(page)
                else:
                    tuned_page = ' '.join(words[:window_size*2])
                    res_pages.append(tuned_page)
        else:
            print('== [empty KB] ==',phrase)
        return res_pages

if __name__ == '__main__':
    rank_list = ImprovedRankList()
    rank_list.parse_config('config/config.ini')
    rank_list.initialize()
    rank_list.run()
    # print(rank_list.get_parameter_list())

