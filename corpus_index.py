from py4j.java_gateway import JavaGateway
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
    index_dir_path = "E:/qiuchi/index/index_clueweb12"
    indri = IndriAPI(index_dir_path)
    contexts = indri.get_context_list('field day',window_size = 50)
    # print(get_context_TFIDF(contexts, indri))
    # print(contexts)

    # print(len(docs))


if __name__ == '__main__':
    main()



# gateway.entry_point.print()             # connect to the JVM
  # create a java.util.Random instance
# number1 = random.nextInt(10)              # call the Random.nextInt method
# number2 = random.nextInt(10)
# print(number1,number2)
# addition_app = gateway.entry_point        # get the AdditionApplication
# addition_app.addition(number1,number2)
