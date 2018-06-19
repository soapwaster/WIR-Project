import nltk as nltk
from nltk.tokenize import word_tokenize as wt
from nltk.corpus import wordnet as wn
import numpy as np
from textblob import TextBlob
import io
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re, math
from collections import Counter
def nounify(adjective):
    """
     :param adjective: the adjective
    :return: set of nouns semantically related to it
    """
    set_of_related_nouns = set()

    for lemma in wn.lemmas(wn.morphy(adjective, pos="a")):
        for related_form in lemma.derivationally_related_forms():
            for synset in wn.synsets(related_form.name(), pos=wn.NOUN):
                set_of_related_nouns.add(synset)

    return set_of_related_nouns

def extract_noun_phrases(txt):
    """
    Extract only noun phrases from a sentence
    :param txt: the sentence
    """
    blob = TextBlob(txt)
    print(blob.noun_phrases)

def load_documents(doc_file):
    """
    :param doc_file: File of documents
    :return: List of documents where the index is the doc-id
    """
    in_file = open(doc_file, "r")
    text = in_file.read()
    in_file.close()

    docs_list = []

    #Add fake element in position 0 since docs begin with 1
    docs_list.append("")


    docs = text.strip().split("/")
    for el in docs:
        el = el.strip()
        el = io.StringIO(el)
        doc_id = el.readline()
        doc = el.read().strip() #Read the rest of the file
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
        term_term_dict[ttv[0]+";"+ttv[1]] = ttv[2]
    in_file.close()
    return term_term_dict

def tokenize(text):
    tokenizer = wt(text)
    return tokenizer

def tf_idf(docs, q_docs):
    """
    Returns the tf-idf matrix. TODO is the fact that the idf's should be independed, but here they are not.
    :param docs: document set (list)
    :param q_docs: query set (list)
    :return: tf-idf matrix
    """
    s_docs = docs + q_docs
    tf = CountVectorizer()
    tfT = TfidfTransformer(use_idf=False, sublinear_tf=True, norm=False)
    tf_matrix = tf.fit_transform(s_docs)
    s_docs_fn = tf.get_feature_names()

    #Tf_matrix
    tf_matrix = tfT.transform(tf_matrix,copy=False)

    #Idf for docs
    idfD = TfidfTransformer(use_idf=True)
    idfD = idfD.fit(tf.fit_transform(docs))
    docs_fn = tf.get_feature_names()
    ones_for_docs = np.ones(len(s_docs_fn)) # [1,1,1,1,1,1,...,1]
    for i,el in enumerate(docs_fn):
        index = s_docs_fn.index(el)
        ones_for_docs[index] = ones_for_docs[index] * idfD.idf_[i]

    tf_matrix[0:4, :] = tf_matrix[0:4, :].multiply(np.tile(ones_for_docs, (4, 1)))

    #Ids for queries
    idfQ = TfidfTransformer(use_idf=True)
    idfQ = idfQ.fit(tf.fit_transform(q_docs))
    query_fn = tf.get_feature_names()
    ones_for_docs = np.ones(len(s_docs_fn))  # [1,1,1,1,1,1,...,1]
    for i,el in enumerate(query_fn):
        index = s_docs_fn.index(el)
        ones_for_docs[index] *= idfQ.idf_[i]

    tf_matrix[4:,:] = tf_matrix[4:,:].multiply(np.tile(ones_for_docs,(len(q_docs)+1,1)))
    return tf_matrix[0:4,:], tf_matrix[4:,:], s_docs_fn
'''print(nounify("apologetic"))
print(nounify("lonely"))
print(nounify("handsome"))'''

def compute_cosine_similarity(docs, queries):
    dd = load_documents(docs)
    qq = load_documents(queries)
    tf, qtf, terms = tf_idf(dd, qq)
    return cosine_similarity(tf, qtf), terms

'''
Computes the new cosine similarity TODO
'''
def new_cosine_similarity(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x] ** 2 for x in vec1.keys()])
    sum2 = sum([vec2[x] ** 2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator


print(compute_cosine_similarity("file.txt", "query.txt")[1])
