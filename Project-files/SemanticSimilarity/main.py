import similarity_utilities as wnu
import load_utilities as lu
import test_functions as test_f
import time

#d = lu.load_documents("Test/Dataset/document.txt")
#q = lu.load_documents("Test/Dataset/query.txt")
#a,b,term = lu.tf_idf(d,q)

tf = lu.load_sparse("Intermidiate-data-structure/tfidf_doc_matrix.npz")
qtf = lu.load_sparse("Intermidiate-data-structure/tfidf_query_matrix.npz")
terms = lu.load_terms("Intermidiate-data-structure/termID_mapping_list.txt")
#al = su.compute_cosine_similarity(tf,qtf)

#wnu.all_term_sim(wnu.custom_similarity,"Intermidiate-data-structure/our_custom_sim.csv")
#wnu.term_sim_compare(lu.load_terms("Test/Dataset/wordsim.csv"),wnu.custom_similarity)
common = test_f.test_query(14,tf,qtf,terms)
print(common)


