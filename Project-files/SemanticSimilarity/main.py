import similarity_utilities as wnu
import load_utilities as lu
import time

#d = lu.load_documents("Test/Dataset/document.txt")
#q = lu.load_documents("Test/Dataset/query.txt")
#a,b,term = lu.tf_idf(d,q)

tf = lu.load_sparse("Intermidiate-data-structure/tfidf_doc_matrix.npz")
qtf = lu.load_sparse("Intermidiate-data-structure/tfidf_query_matrix.npz")
terms = lu.load_terms("Intermidiate-data-structure/termID_mapping_list.txt")
al = su.compute_cosine_similarity(tf,qtf)

#wnu.all_term_sim(wnu.custom_similarity,"Intermidiate-data-structure/our_custom_sim.csv")
#wnu.term_sim_compare(lu.load_terms("Test/Dataset/wordsim.csv"),wnu.custom_similarity)
ll = lu.load_all_relevant_docs("Test/Dataset/rlv-ass.txt")
ff = wnu.query_sims(qtf[8],tf,terms,wnu.custom_similarity,ll)
tot = time.time()-start
print("time elapsed  opt : "+str(tot))

#wnu.top_relevant_docs_for_query_from_cosine(tf,qtf,"Intermidiate-data-structure/rlv-cosine.txt")

dd = lu.load_documents("Test/Dataset/rlv-ass.txt")
print(dd[7])
gg = []
for j in range(1, 98):
    gg.append(str(ff[j][1]))

print(dd[7].split(" "))
print(gg)
c = list(set(dd[7].split(" ")).intersection(gg))
print(c)

