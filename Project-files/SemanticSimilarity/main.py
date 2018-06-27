import wordnet_utils as wnu
import time
'''
start = time.time()
s1=no_opt_gvsm_similarity_Approx1_qq_dd_dq(d1,q1,terms,sim)
tot = time.time()-start
print("time elapsed non opt : "+str(tot))
print("total estimated : "+str(tot*93*11429))
'''
d = wnu.load_documents("file.txt")
q = wnu.load_documents("query.txt")
a,b,terms = wnu.tf_idf(d,q)

tf = wnu.load_sparse("tfidf_doc_matrix.npz")
qtf = wnu.load_sparse("tfidf_query_matrix.npz")
start = time.time()
#s2=gvsm_similarity_Approx2_dq(d1,q1,terms,f)
#s2=opt_gvsm_similarity_Approx1_qq_dd_dq(d1,q1,terms,sim)
s2=wnu.gvsm_similarity_Approx2_dq(tf[4720],qtf[28],terms,wnu.f)
s3=wnu.gvsm_similarity_Approx2_dq(tf[5472],qtf[28],terms,wnu.f)
s4=wnu.gvsm_similarity_Approx2_dq(tf[5583],qtf[28],terms,wnu.f)
s5=wnu.gvsm_similarity_Approx2_dq(tf[5850],qtf[28],terms,wnu.f)
s6=wnu.gvsm_similarity_Approx2_dq(tf[7482],qtf[28],terms,wnu.f)
s7=wnu.gvsm_similarity_Approx2_dq(tf[1],qtf[28],terms,wnu.f)
al = wnu.compute_cosine_similarity(tf,qtf)
#aa = wnu.lowest_common_hypernym("corgi","dog")
#print(list(aa.closure(lambda s: s.hypernyms())))
print(al[4720][28])
print(s2)
print(s3)
print(s4)
print(s5)
print(s6)
print(s7)
tot = time.time()-start
print("time elapsed  opt : "+str(tot))
#print("total estimated : "+str(tot*93*1000))




#print("Non-opt : "+str(s1))
#print("Opt : "+str(s2))

#s3=gvsm_similarity_Approx2_dq(d1,q1,terms,sim)
#print(s3)

