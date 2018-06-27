import wordnet_utils

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

a=TfidfVectorizer(norm=False, use_idf=False, sublinear_tf=True)

b=TfidfTransformer(use_idf=False, sublinear_tf=True, norm=False)

