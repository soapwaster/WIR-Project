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
import scipy
from scipy.sparse import csr_matrix, find


def nounify(adjective):
    """
     :param adjective: the adjective
    :return: set of nouns semantically related to it
    """
    set_of_related_nouns = set()

    for lemma in wn.lemmas(wn.morphy(adjective, pos="a")):
        if len(lemma.derivationally_related_forms()) == 0:
            for related_form in wn.synsets(lemma.name()):
                for synset in related_form.attributes():
                    set_of_related_nouns.add(synset)
        else:
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


def tokenize(text):
    tokenizer = wt(text)
    return tokenizer


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
    tf_matrix[0:4, :] = tf_matrix[0:4, :].multiply(np.tile(ones_for_docs, (4, 1)))

    # tfidf for the queries
    tf_matrix[4:, :] = tf_matrix[4:, :].multiply(np.tile(ones_for_docs, (len(q_docs) + 1, 1)))

    # returns also the mapping termid->term as index->s_docs_fn[index]



    return tf_matrix[0:4, :], tf_matrix[4:, :], s_docs_fn


'''
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
'''

'''print(nounify("apologetic"))
print(nounify("lonely"))
print(nounify("handsome"))'''


def compute_cosine_similarity(docs, queries):
    dd = load_documents(docs)
    qq = load_documents(queries)
    tf, qtf, terms = tf_idf(dd, qq)
    return cosine_similarity(tf, qtf), terms


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


def lowest_common_hypernym(term1, term2):
    synsets_t1 = wn.synsets(term1)
    synsets_t2 = wn.synsets(term2)
    lch = ""

    for s in synsets_t1:
        if s.pos() == 'v':
            continue
        elif s.pos() == 'a':
            synsets_t1.extend(list(nounify(s.lemmas()[0].name())))
            continue
        for t in synsets_t2:
            if t.pos() == 'v':
                continue
            elif t.pos() == 'a':
                synsets_t2.extend(list(nounify(t.lemmas()[0].name())))
                continue
            print(s)
            print(t)
            print(s.lowest_common_hypernyms(t))
            print("-------------")

            lch = s.lowest_common_hypernyms(t)

    return lch


# print(compute_cosine_similarity("file.txt", "query.txt")[1])


'''
print(tf)
print(qtf)
print(terms)
'''

def gvsm_similarity(doc, query, term, similarity):
    #TODO : OPTIMIZATION
    #    r,c,v = find(doc)
    #    r2, c2, v2 = find(query)
    #iterate over the union
    """
    Returns the similarity score
    :param doc: tdidf document vector
    :param query: tdidf query vector
    :param term: array of mapping term
    :param similarity: similarity function
    :return: gvsm score
    """
    dim = len(term)
    score = 0;
    den_doc = 0;
    den_query = 0;




    for i in range(dim):
        d_i = doc[0,i]
        q_i = query[0,i]
        for j in range(i,dim):
            sim = similarity(term[i],term[j])
            docij = ( d_i + doc[0,j] )*sim
            queryij =( q_i + query[0,j] )*sim
            score += ( docij  ) * ( queryij )
            den_doc += np.square(docij)
            den_query += np.square(queryij)

    norm = np.sqrt(den_doc*den_query)
    return score/norm

def f(a,b):
    return 1



dd = load_documents("file.txt")
qq = load_documents("query.txt")
tf, qtf, terms = tf_idf(dd, qq)

print(terms)

d1 = tf[1]
q1 = qtf[2] #query n.1


for i in range(len(terms)):
    if (q1[0,i]!=0): print(terms[i])


#print(q1)
import time
start = time.time()
s=gvsm_similarity(d1,q1,terms,f)
tot = time.time()-start
print(s)
print("time elapsed : "+str(tot))
print("total estimated : "+str(tot*93*11429))
#print(tf[2:3,:])
#print(qtf)
#print(len(terms))
#print(terms[416])

#d1 = tf[1]
#print(d1)
#q1 = qtf[1]

#print(d1[0,416])

#print("------------------------\n")
#print(d1[0])

#s=gvsm_similarity(tf[1],qtf[1],terms,f)
#print("------------------------\n")
#print(s)

'''

cx = scipy.sparse.coo_matrix()
print(cx.getrow(1))
print("------------------------")
for i,j,v in zip(cx.row, cx.col, cx.data):
    print("(%d, %d), %s" % (i,j,v))
'''

