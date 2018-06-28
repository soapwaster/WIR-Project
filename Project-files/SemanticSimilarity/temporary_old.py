import wordnet_utils_old

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
from nltk.corpus import wordnet
import sys
document_0 = "Tree Sun"
document_1 = "Ice Tree"
document_2 = "House Garden"
document_3 = "Good Bad"
document_4 = "Pool Ball"

query_0 = "Ice Tree"
all_documents = [document_0, document_1,document_2,document_3,document_4]
all_queries =[query_0]

from scipy.sparse import coo_matrix
from sklearn.feature_extraction.text import TfidfVectorizer


def tfidfNEW(docs,qdocs):

    sklearn_tfidf = TfidfVectorizer(norm='l2',min_df=0, use_idf=True, smooth_idf=False, sublinear_tf=True, tokenizer=wordnet_utils.tokenize)
    doc_vect = sklearn_tfidf.fit_transform(docs)
    term_index = sklearn_tfidf.get_feature_names()
    print(term_index)

    return doc_vect, term_index, sklearn_tfidf.idf_

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
            #ris.append((b[i2], val_d[i1], 0))
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







md = [[4,5,6,7,8,9,0,0,2]]


mq =[[4,50,78,89,450,9000,5,6,0]]

doc = scipy.sparse.csr_matrix(md)
query = scipy.sparse.csr_matrix(mq)

term = ["casa","albero","vacca","casa","albero","vacca","casa","albero","vacca"]
def f(a,b):
    return 1


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

    print("no opt")



    tot = merge(col_d,col_q)

    print("merged no opt "+str(tot))


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
            #print(" docij queryij " + " " + str(docij) + " " + str(queryij))
            print(" docij : "+str(docij)+" queryij : "+str(queryij)+" product : "+str(( docij  ) * ( queryij )))
            print("score " + str(score))


    norm = np.sqrt(den_doc*den_query)

    return score/norm


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



def gvsm_approx_similarity_AAAA(doc, query, term, similarity):

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


    i=0
    j=0

    flag1 = False
    flag2 = False
    flag3 = False
    z = 0

    while(i<dim):
        termID_ith = tot[i][0]



        d_i =  tot[i][1]
        q_i = tot[i][2]
        #print(term[termID_ith])
        #print(" d_i q_i " + " " + str(d_i) + " " + str(d_i))
        while (j+i<dim):



            #print(i)
            #print(j)
            termID_jth = tot[j+i][0]

            #print(term[termID_jth])

            sim = similarity(term[termID_ith],term[termID_jth])
            docij = ( d_i + tot[j+i][1] )*sim
            queryij =( q_i + tot[j+i][2]  )*sim




            score += ( docij  ) * ( queryij )



            #print(" docij : " + str(docij) + " queryij : " + str(queryij) + " product : " + str((docij) * (queryij)))
            #print("score "+str(score))
            den_doc += np.square(docij)
            den_query += np.square(queryij)

            if (flag2):
                j+=1
                flag2= False
                #if ( not flag3 ): flag3 = False
                continue
            if (flag1):
                j+=1
                flag3 = True
                flag1= False
                continue

            if (tot[j+i][1]==0 or tot[j+i][2]==0 ):
                j+=1

            else :
                flag2 = True
        j=0
        if (flag3):
            flag3=False

            i+=1
            continue



        if (d_i == 0 or q_i == 0):
            i+=1
        else :
            flag1 = True


    norm = np.sqrt(den_doc*den_query)

    return score/norm

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

s1=no_opt_gvsm_similarity_Approx1_qq_dd_dq(doc,query,term,f)

s2=gvsm_approx_similarity_AAAA(doc,query,term,f)

print("no-opt vs opt "+str(s1)+" "+str(s2))