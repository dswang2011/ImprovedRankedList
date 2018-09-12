import os
import io,re,codecs
import numpy as np
import configparser
import argparse
from corpus_index import IndriAPI
from word_embedding import form_matrix
from knowledge_base import get_prepared_KB
from utils import *
import math

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
        self.knowledge_base = get_prepared_KB(self.knowledge_base_path)
        self.corpus_index = IndriAPI(self.index_dir_path)
        self.word_embedding = None
        if 'path_to_vec' in self.__dict__:
            self.word_embedding = form_matrix(self.path_to_vec)

    def run(self):
        # self.rank_list_comp('field day', ' A day of class taken away from school for a field trip.')
        phrases,scenarios,labels = read_test_data(self.input_path)
        # print(phrases)
        # print(scenarios)
        output_file = 'output.txt'
        if 'output_file' in self.__dict__:
            output_file = self.output_file

        file_writer = codecs.open(output_file,'w')
        scores = []
        targets = []
        i = 0
        for phrase,scenario,label in zip(phrases,scenarios,labels):
            if i >0:
                score = self.rank_list_comp(phrase, scenario)
                print(score)
                scores.append(score)
                targets.append(float(label))
                file_writer.write('{}\t{}\t{}\t{}\n'.format(phrase,scenario,score,label))
            i = i+1
        pearson_correlation_coefficient = pearson_correlation(scores,targets)
        writer = codecs.open(self.pearson_correlation_file,'w')
        writer.write(str(pearson_correlation_coefficient))
        print(pearson_correlation_coefficient)


     # def initialize(self):
    def rank_list_comp(self, p, scenario):
        #scenario = stem_words(scenario)
        p_stem = stem_words(p)
        phrase_context_terms = self.get_window_list(p)

        print('Computing the context representation.')
        phrase_context_rep = self.get_context_rep(phrase_context_terms)
        if self.adapt_with_knowledge_base:
            print('Recomputing the context representation with Knowledge base.')
            # Get phrase diambiguation pages in the knowledge base
            candidate_context_list = self.get_candidate_pages(p_stem)
            # Find all (matched contexts, matching scores) above a given threshold
            matched_contexts_dic = self.get_matched_contexts(scenario, candidate_context_list)

            # Generate the contexts of the original phrase based on matched candidate pages
            phrase_context_rep = self.compute_updated_context(phrase_context_rep, matched_contexts_dic)

        print('Get perturbed phrase list.')
        perturbed_phrase_list = get_perturbed_phrases(p)

        print('Compute output score for determining CD/NCD.')
        output_score = 0
        # get context of perturbed phrases
        for perturbed_phrase in perturbed_phrase_list:
            stem_perturb = stem_words(perturbed_phrase)
            context_list = self.get_window_list(stem_perturb)
            context_rep = self.get_context_rep(context_list)
            output_score = output_score + self.get_context_similarity(phrase_context_rep, context_rep)
        avg_perturb_score = 0 # default CD score
        if len(perturbed_phrase_list)>0:
            avg_perturb_score = output_score/len(perturbed_phrase_list)
        if avg_perturb_score ==0:
            print('00:',p,perturbed_phrase_list)
        return avg_perturb_score

    # get window list for a phrase (or perturbs)
    def get_window_list(self, perturbed_phrase):
        window_size = 10
        if 'window_size' in self.__dict__:
            window_size = self.window_size
        context_list = self.corpus_index.get_context_list(perturbed_phrase, window_size = window_size)
        return context_list

    def get_context_rep(self, context_list):
        context_rep = None
        if self.context_type == 'tfidf':
            collection_doc_count = self.corpus_index.get_collection_doc_count()
            context_rep = {}
            for context in context_list:
                term_list = context.split()
                for term in term_list:
                    idf = math.log10(collection_doc_count/(self.corpus_index.get_doc_frequency(term)+1))
                    if term in context_rep:
                        context_rep[term] = context_rep[term]+idf
                    else:
                         context_rep[term] = idf
            for term in context_rep:
                context_rep[term] = context_rep[term]/len(context_list)
        elif self.context_type == 'word_embedding':
            matrix, word_list = self.word_embedding
            context_rep = np.zeros(shape = (1,matrix.shape[1]))
            for context in context_list:
                term_list = context.split()
                # term_vector
                index_list = []
                for term in term_list:
                    if term in word_list:
                        index_list.append(word_list.index(term))
                context_mean = np.mean(matrix[index_list,:],axis = 0)
                context_rep = context_rep + context_mean
        # print(context_vector.shape)
            context_rep = context_rep/len(context_list)   # 300 dimention
        return context_rep

    # 300 dimension vect
    def get_vect_rep(self, context_list):
        context_rep = None
        matrix, word_list = self.word_embedding
        context_rep = np.zeros(shape = (1,matrix.shape[1]))
        for context in context_list:
            term_list = context.split()
            # term_vector
            index_list = []
            for term in term_list:
                if term in word_list:
                    index_list.append(word_list.index(term))
            context_mean = np.mean(matrix[index_list,:],axis = 0)
            context_rep = context_rep + context_mean
        # print(context_vector.shape)
        context_rep = context_rep/len(context_list)   # 300 dimention
        return context_rep

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

            for term in context_rep_2:
                sum_square_weight_2 =  sum_square_weight_2+ context_rep_2[term]**2
            similarity_score = inner_product/(np.sqrt(sum_square_weight_1*sum_square_weight_2)+0.0001)

        elif self.context_type == 'word_embedding':
            similarity_score = np.inner(context_rep_1, context_rep_2)/(np.linalg.norm(context_rep_1)*np.linalg.norm(context_rep_2))
        return similarity_score

    def get_vect_similarity(self, vect_rep_1,vect_rep_2):
        similarity_score = 0
        similarity_score = np.inner(vect_rep_1, vect_rep_2)/(np.linalg.norm(vect_rep_1)*np.linalg.norm(vect_rep_2))
        return similarity_score


    def get_matched_contexts(self, phrase_senario, candidate_contexts_list):
        phrase_vect_rep = self.get_vect_rep([phrase_senario])
        threshold = 0
        if 'kb_matching_threshold' in self.__dict__:
            threshold = self.kb_matching_threshold
        matched_contexts_pairs = []
        total_score = 0
        for candidate_context in candidate_contexts_list:
            candidate_context_rep = self.get_context_rep([candidate_context])
            candidate_vect_rep = self.get_vect_rep([candidate_context])
            score = self.get_vect_similarity(phrase_vect_rep, candidate_vect_rep)
            if score > threshold:
                matched_contexts_pairs.append((candidate_context_rep,score))
                # matched_contexts_dic[candidate_context] = score
                total_score = total_score + score

        # Normalize similarity scores
        for i in range(len(matched_contexts_pairs)):
            matched_contexts_pairs[i] = (matched_contexts_pairs[i][0], matched_contexts_pairs[i][1]/total_score)

        return matched_contexts_pairs

    def compute_updated_context(self,phrase_context,matched_contexts_pairs):
        weight = 0.1
        if 'kb_combine_weight' in self.__dict__:
            weight = self.kb_combine_weight
        output_context = phrase_context.copy()
        if self.context_type == 'tfidf':
            for term in output_context:
                output_context[term] = output_context[term]* weight
        elif self.context_type == 'word_embedding':
            output_context = output_context * weight

        for context_rep,score in matched_contexts_pairs:
            if self.context_type == 'tfidf':
                for term in context_rep:
                    if term in output_context:
                        output_context[term] = output_context[term] + context_rep[term]*score* (1-weight)
                    else:
                        output_context[term] = context_rep[term]*score* (1-weight)
            elif self.context_type == 'word_embedding':
                output_context = output_context + (1-weight)*score*context_rep

        return output_context

    def get_candidate_pages(self,phrase):
        if phrase.strip() in self.knowledge_base:
            return self.knowledge_base[phrase]
        else:
            return []

if __name__ == '__main__':
    rank_list = ImprovedRankList()
    rank_list.parse_config('config/config.ini')
    rank_list.initialize()
    rank_list.run()
    # print(rank_list.get_parameter_list())

