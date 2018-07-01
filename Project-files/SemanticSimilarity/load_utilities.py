from nltk.tokenize import word_tokenize as wt
from nltk.corpus import wordnet as wn
import numpy as np
from textblob import TextBlob
import io
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import scipy
from scipy.sparse import csr_matrix, find
from nltk.corpus import wordnet
import io

import numpy as np
import scipy
from nltk.corpus import wordnet
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize as wt
from scipy.sparse import csr_matrix, find
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob


def load_documents(doc_file):
    """
    :param doc_file: File of documents
    :return: List of documents where the index is the doc-id
    """
    s = 0
    in_file = open(doc_file, "r")
    text = in_file.read()
    in_file.close()

    docs_list = []

    # Add fake element in position 0 since docs begin with 1
    docs_list.append("")

    docs = text.strip().split("/")
    for el in docs:
        el = el.strip()
        el = io.StringIO(el)
        doc_id = el.readline()
        doc = el.read().strip()  # Read the rest of the file
        docs_list.append(doc.lower())
    return docs_list


def load_terms(term_term_file):
    """
    Return a dict with term1;term2 as key and the gold standard as value
    :param term_term_file: the term-term file with the golden standard
    :return: dict
    """
    in_file = open(term_term_file, "r")

    term_term_dict = {}

    for line in in_file:
        line = line.strip()
        ttv = line.split(";")
        term_term_dict[ttv[0] + ";" + ttv[1]] = ttv[2]
    in_file.close()
    return term_term_dict


def tf_idf(docs, q_docs):
    """
    Returns the tf-idf matrix. TODO is the fact that the idf's should be independed, but here they are not.
    :param docs: document set (list of string)
    :param q_docs: query set (list of string)
    :return: tf-idf matrix
    """
    s_docs = docs + q_docs
    tf = CountVectorizer(lowercase=True, stop_words='english', strip_accents='unicode',
                         tokenizer=tokenize)  # To build the count matrix
    tfT = TfidfTransformer(use_idf=False, sublinear_tf=True,
                           norm=False)  # To build the count sublinear matrix . The fit_transform require a count matrix obtianed by CountVectorizer

    tf_matrix = tf.fit_transform(s_docs)
    s_docs_fn = tf.get_feature_names()  # All the terms in documents and query with the index as id

    # Tf_matrix
    tf_matrix = tfT.transform(tf_matrix, copy=False)  # The tf  matrix wrt documents and queries
    # Idf for docs
    idfD = TfidfTransformer(use_idf=True)  # To get the idf for only the terms of the documents

    idfD = idfD.fit(tf.fit_transform(docs))
    docs_fn = tf.get_feature_names()
    # ones_for_docs = np.ones(len(s_docs_fn)) # [1,1,1,1,1,1,...,1]

    # At the i-th position there is the idf of the i-th term according the mapping of s_docs_fn
    # It must be initilized to 1+ln(N=#documents) that is the idf when a term of the query is not present in the vocavulary of the documents
    ones_for_docs = np.full(len(s_docs_fn), 1 + np.log(len(docs_fn)))
    for i, el in enumerate(docs_fn):
        index = s_docs_fn.index(el)
        # ones_for_docs[index] = ones_for_docs[index] * idfD.idf_[i]
        ones_for_docs[index] = idfD.idf_[i]

    # tfidf for the documents
    for i in range(0,len(docs)):
        tf_matrix[i, :] = tf_matrix[i, :].multiply(ones_for_docs)

    # tfidf for the queries
    for i in range(len(docs),len(docs)+len(q_docs)):
        tf_matrix[i, :] = tf_matrix[i, :].multiply(ones_for_docs)


    save_sparse(tf_matrix[0:len(docs), :],"Intermidiate-data-structure/tfidf_doc_matrix")
    save_sparse(tf_matrix[len(docs):, :],"Intermidiate-data-structure/tfidf_query_matrix")
    save_terms("Intermidiate-data-structure/termID_mapping_list.txt",s_docs_fn)
    # returns also the mapping termid->term as index->s_docs_fn[index]
    return tf_matrix[0:len(docs), :], tf_matrix[len(docs):, :], s_docs_fn

def save_sparse(sp_m,name):
    scipy.sparse.save_npz(name+".npz", sp_m)
def load_sparse(file_name):
    return scipy.sparse.load_npz(file_name)

def save_terms(file_name,array):
    scipy.savetxt(file_name,array,fmt='%s')

def load_terms(file_name):
    return scipy.loadtxt(file_name,dtype='str')


def tokenize(text):
    tokenizer = wt(text)
    return tokenizer

def load_all_relevant_docs(file):
    """
    :param file: relevance file
    :return: set of relevant docs
    """
    rel_list= set()
    rel_docs = load_documents(file)
    for el in rel_docs:
        v = el.strip().split(" ")
        for el in v:
            el = el.strip().split("\n")
            for ul in el:
                if ul != "":
                    rel_list.add(int(ul))


    return rel_list

def load_relevant_for_query(query_id, file):
    real_rel = load_documents("Test/Dataset/rlv-ass.txt")[query_id]
    rel_list = []

    v = real_rel.strip().split(" ")
    for el in v:
        el = el.strip().split("\n")
        for ul in el:
            if ul != "":
                rel_list.append(int(ul))
    return rel_list

def save_query_intersection(query_id, query_list):
    file = open("Test/Relevance/relevance_"+str(query_id)+".txt", "w")
    file.write(str(query_id))
    file.write("\n")
    for el in query_list:
        file.write(str(el) + ", ")
    file.write("\n/")
    file.close()