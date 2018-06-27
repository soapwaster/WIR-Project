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


def no_opt_gvsm_similarity_Approx1_qq_dd_dq(doc, query, term, similarity):
    '''
    TODO: optimization
    Accessing to the tfidf score in the sparse vector doc or query is expensive .
    The optimization can be done becuase find returns the tfidf score associated to a doc or query

    '''

    """
    Returns the approximated similarity score iterating over the union of terms in doc and query
    :param doc: tdidf document vector
    :param query: tdidf query vector
    :param term: array of mapping term
    :param similarity: similarity function
    :return: gvsm score
    """

    row_d,col_d,val_d = find(doc)
    row_q, col_q,val_q = find(query)


    tot = merge(col_d,col_q)


    dim = len(tot)
    score = 0;
    den_doc = 0;
    den_query = 0;




    for i in range(dim):
        d_i = doc[0,tot[i]] #very expensive
        q_i = query[0,tot[i]] #very expensive
        for j in range(i,dim):
            sim = similarity(term[tot[i]],term[tot[j]])
            docij = ( d_i + doc[0,tot[j]] )*sim #very expensive
            queryij =( q_i + query[0,tot[j]] )*sim #very expensive
            score += ( docij  ) * ( queryij )
            den_doc += np.square(docij)
            den_query += np.square(queryij)

    norm = np.sqrt(den_doc*den_query)

    return score/norm

def gvsm_similarity_Approx2_dq(doc, query, term, similarity):

    """
    Returns the similarity score approximated iterating over the doc and query terms but it should be optimized
    :param doc: tdidf document vector
    :param query: tdidf query vector
    :param term: array of mapping term
    :param similarity: similarity function
    :return: gvsm score
    """
    row_d,col_d,val_d = find(doc)
    row_q, col_q, val_q = find(query)
    print(col_d)
    print(col_q)

    #tot = merge(col_d,col_q)

    dim_doc = len(col_d)
    dim_query = len(col_q)
    score = 0;
    den_doc = 0;
    den_query = 0;




    for i in range(dim_doc):
        d_i = doc[0,col_d[i]]
        q_i = query[0,col_d[i]]
        for j in range(dim_query):
            sim = similarity(term[col_d[i]],term[col_q[j]])
            docij = ( d_i + doc[0,col_q[j]] )*sim
            queryij =( q_i + query[0,col_q[j]] )*sim
            score += ( docij  ) * ( queryij )
            den_doc += np.square(docij)
            den_query += np.square(queryij)

    norm = np.sqrt(den_doc*den_query)
    return score/norm



def gvsm_similarity_complete_slow(doc, query, term, similarity):
    """
    Returns the correct similarity score but it is too expensive
    :param doc: tdidf document vector
    :param query: tdidf query vector
    :param term: array of mapping term
    :param similarity: similarity function
    :return: gvsm score
    """
    #row_d,col_d,val_d = find(doc)
    #row_q, col_q, val_q = find(query)
    #print(col_d)
    #print(col_q)



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
    s = 0
    in_file = open(doc_file, "r")
    text = in_file.read()
    in_file.close()

    docs_list = []

    # Add fake element in position 0 since docs begin with 1
    docs_list.append("")

    docs = text.strip().split("/")
    for el in docs:
        s = s +1
        print(str(s) + "")
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





def compute_cosine_similarity(tfidfdocs, tfidfqueries):
    return cosine_similarity(tfidfdocs, tfidfqueries)

def compute_cosine_similarity_from_file(tf, qtf):
    dd = load_documents(tf)
    qq = load_documents(qtf)

    a,b,c = tf_idf(dd,qq)
    return compute_cosine_similarity(a,b)

def lowest_common_hypernym(term1, term2):
    synsets_t1 = wn.synsets(term1)
    synsets_t2 = wn.synsets(term2)
    max_depth = 0
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
            lc = s.lowest_common_hypernyms(t)
            depth = lc[0].min_depth()
            if lc != [] and  depth > max_depth:
                max_depth = depth
                lch = lc[0]
                #lch.append(s.lowest_common_hypernyms(t))


    return lch


def gvsm_approx_similarity(doc, query, term, similarity):

    """
    Returns the approximated similarity score iterating over the union of terms in doc and query
    :param doc: tdidf document vector
    :param query: tdidf query vector
    :param term: array of mapping term
    :param similarity: similarity function
    :return: gvsm score
    """

    row_d,col_d,val_d = find(doc)
    row_q, col_q,val_q = find(query)





    tot = super_merge(col_d,col_q,val_d,val_q)
    #tot is an array of triples (term_id , tfidf_termID in doc , tfidf_termID in query) sorted by term_id


    dim = len(tot)
    score = 0;
    den_doc = 0;
    den_query = 0;



    for i in range(dim):
        termID_ith = tot[i][0]


        d_i =  tot[i][1]
        q_i = tot[i][2]
        #print(term[termID_ith])
        for j in range(i,dim):
            #print(i)
            #print(j)
            termID_jth = tot[j][0]

            #print(term[termID_jth])

            sim = similarity(term[termID_ith],term[termID_jth])
            docij = ( d_i + tot[j][1] )*sim
            queryij =( q_i + tot[j][2]  )*sim
            score += ( docij  ) * ( queryij )
            #print("terms : "+str(term[termID_ith])+" "+str(term[termID_jth])+" docij : "+str(docij)+" queryij : "+str(queryij)+" product : "+str(( docij  ) * ( queryij )))
            den_doc += np.square(docij)
            den_query += np.square(queryij)

    norm = np.sqrt(den_doc*den_query)

    return score/norm



def super_merge(a,b,val_d,val_q):
    '''

    :param a: array of integer : termid_doc
    :param b: array of integer : termid_query
    :param val_d: array of integer : val_doc
    :param val_q: array of integer : val_query
    :return: merged sorted array  by termid, of triples (termid,val_doc,val_query)
    '''


    i1=0
    i2=0
    la = len(a)
    lb = len(b)
    lab = la+lb
    ris = []
    for t in range(lab):
        if (i1+i2>lab-1): break;
        if (i1>la-1):
            ris.append( (b[i2],0,val_q[i2]))
            i2+=1
        elif (i2>lb-1) :
            ris.append((a[i1],val_d[i1],0))
            i1+=1

        elif (a[i1]<b[i2]):
            ris.append((a[i1],val_d[i1],0))
            i1+=1;

        elif (b[i2]<a[i1]):
            ris.append((b[i2],0,val_q[i2]))
            i2+=1;


        else :

            ris.append((b[i2],val_d[i1],val_q[i2]))
            i2+=1
            i1+=1


    return ris

def merge(a,b):

    i1=0
    i2=0
    la = len(a)
    lb = len(b)
    lab = la+lb
    ris = []
    for t in range(lab):
        if (i1+i2>lab-1): break;
        if (i1>la-1):
            ris.append( b[i2])
            i2+=1
        elif (i2>lb-1) :
            ris.append(a[i1])
            i1+=1

        elif (a[i1]<b[i2]):
            ris.append(a[i1])
            i1+=1;

        else :
            ris.append(b[i2])
            i2+=1;


    return ris






def save_sparse(sp_m,name):
    scipy.sparse.save_npz(name+".npz", sp_m)
def load_sparse(file_name):
    return scipy.sparse.load_npz(file_name)

def save_terms(file_name,array):
    scipy.savetxt(file_name,array,fmt='%s')

def load_terms(file_name):
    return scipy.loadtxt(file_name,dtype='str')

def f(a,b):
    return 1



def sim(word1,word2):

    s=0



    l1=wordnet.synsets(word1, pos="n")
    l2=wordnet.synsets(word2, pos="n")


    if (l1 and l2):
        s = l1[0].wup_similarity(l2[0])
        #print(str(l1[0]) + " ---- " + str(l2[0]) + " simi : " + str(s))
    if (s is None):
        s=0



    return s