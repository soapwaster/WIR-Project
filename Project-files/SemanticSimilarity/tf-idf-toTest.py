import wordnet_utils


document_0 = "Tree Sun"
document_1 = "Ice Tree"

query_0 = "Ice House"
all_documents = [document_0, document_1]
all_queries =[query_0]

from scipy.sparse import coo_matrix
from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(docs, q_docs):
    sklearn_tfidf = TfidfVectorizer(norm='l2',min_df=0, use_idf=True, smooth_idf=False, sublinear_tf=True, tokenizer=wordnet_utils.tokenize)
    doc_query_vect = sklearn_tfidf.fit_transform(docs+q_docs)
    term_index = sklearn_tfidf.get_feature_names();
    return doc_query_vect[0: len(docs),:],doc_query_vect[len(docs),:],term_index

a,b,c = tf_idf(all_documents,all_queries)


print("Document :\n")
cx1 = coo_matrix(a)
for doc, term, val in zip(cx1.row, cx1.col, cx1.data):
    print("The tf-idf of term : " + str(c[term]) + " and docId : " + str(doc) + " is " + str(val))

print("\nQuery :\n")
cx2 = coo_matrix(b)
for doc, term, val in zip(cx2.row, cx2.col, cx2.data):
    print("The tf-idf of term : " + str(c[term]) + " and queryId : " + str(doc) + " is " + str(val))