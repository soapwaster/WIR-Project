import wordnet_utils


document_0 = "Tree Sun"
document_1 = "Ice Tree"

query_0 = "Ice Tree"
all_documents = [document_0, document_1]
all_queries =[query_0]

from scipy.sparse import coo_matrix
from sklearn.feature_extraction.text import TfidfVectorizer


def tfidf(docs):

    sklearn_tfidf = TfidfVectorizer(norm='l2',min_df=0, use_idf=True, smooth_idf=False, sublinear_tf=True, tokenizer=wordnet_utils.tokenize)
    doc_vect = sklearn_tfidf.fit_transform(docs)
    term_index = sklearn_tfidf.get_feature_names()
    print(term_index)

    return doc_vect, term_index, sklearn_tfidf.idf_



def similarity(doc_vect,doc_index,idf_scores,query):
    # doc_vect is a matrix docid-termid
    # doc_index is a dictonary termID:index
    #idf_scores is a list of idf
    #query is a vector
    from collections import Counter
    from sklearn import preprocessing
    sklearn_tfidf = TfidfVectorizer( min_df=0, use_idf=True, smooth_idf=False, sublinear_tf=True,
                                    tokenizer=wordnet_utils.tokenize)

    counts = sklearn_tfidf.fit_transform(query)

    pos = 0

    print(doc_index)
    print(counts)

    for key in keys:
        if ( doc_index.get(key) ):

            query_vec[pos] = counts.get(key)*idf_scores[  doc_index.get(key)  ]
        else :
            query_vec[pos] = counts.get(key)
        pos+=1
    #query_vec = preprocessing.normalize(query_vec,norm='l2')

    return query_vec



a,b,c = tfidf(all_documents)

dict = {k: v for v, k in enumerate(b)}

#similarity(a,dict,c,[query_0])











'''



a,b,c = tfidf(all_documents)

print(c);

print("Document :\n")
cx1 = coo_matrix(a)
for doc, term, val in zip(cx1.row, cx1.col, cx1.data):
    print("The tf-idf of term : " + str(b[term]) + " and docId : " + str(doc) + " is " + str(val))


'''