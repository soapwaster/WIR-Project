import wordnet_utils


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


def tfidf(docs):

    sklearn_tfidf = TfidfVectorizer(norm='l2',min_df=0, use_idf=True, smooth_idf=False, sublinear_tf=True, tokenizer=wordnet_utils.tokenize)
    doc_vect = sklearn_tfidf.fit_transform(docs)
    term_index = sklearn_tfidf.get_feature_names()
    print(term_index)

    return doc_vect, term_index, sklearn_tfidf.idf_

d,q,t = wordnet_utils.tf_idf(all_documents,all_queries)

print(d)
print(q)
print(t)