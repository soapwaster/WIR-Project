import similarity_utilities as wnu
import load_utilities as lu
import time

#d = lu.load_documents("Test/Dataset/document.txt")
#q = lu.load_documents("Test/Dataset/query.txt")
#a,b,term = lu.tf_idf(d,q)

tf = lu.load_sparse("Intermidiate-data-structure/tfidf_doc_matrix.npz")
qtf = lu.load_sparse("Intermidiate-data-structure/tfidf_query_matrix.npz")
terms = lu.load_terms("Intermidiate-data-structure/termID_mapping_list.txt")
#print(terms)

def printdiff(a,b):
    print(str(a) + "---" + str(b))
'''
start = time.time()
s1=no_opt_gvsm_similarity_Approx1_qq_dd_dq(d1,q1,terms,sim)
tot = time.time()-start
print("time elapsed non opt : "+str(tot))
print("total estimated : "+str(tot*93*11429))

d = wnu.load_documents("file.txt")
q = wnu.load_documents("query.txt")
a,b,terms = wnu.tf_idf(d,q)

tf = wnu.load_sparse("tfidf_doc_matrix.npz")
qtf = wnu.load_sparse("tfidf_query_matrix.npz")
start = time.time()
#s2=gvsm_similarity_Approx2_dq(d1,q1,terms,f)
#s2=opt_gvsm_similarity_Approx1_qq_dd_dq(d1,q1,terms,sim)
s2=su.no_opt_gvsm_similarity_Approx1_qq_dd_dq(tf[4720],qtf[28],terms,su.f)
s3=su.no_opt_gvsm_similarity_Approx1_qq_dd_dq(tf[5472],qtf[28],terms,su.f)
s4=su.no_opt_gvsm_similarity_Approx1_qq_dd_dq(tf[5583],qtf[28],terms,su.f)
s5=su.no_opt_gvsm_similarity_Approx1_qq_dd_dq(tf[5850],qtf[28],terms,su.f)
s6=su.no_opt_gvsm_similarity_Approx1_qq_dd_dq(tf[7482],qtf[28],terms,su.f)
s7=su.no_opt_gvsm_similarity_Approx1_qq_dd_dq(tf[1],qtf[28],terms,su.f)

print(s2)
print(s3)
print(s4)
print(s5)
print(s6)
print(s7)

s2=su.gvsm_approx_similarity(tf[4720],qtf[28],terms,su.f)
s3=su.gvsm_approx_similarity(tf[5472],qtf[28],terms,su.f)
s4=su.gvsm_approx_similarity(tf[5583],qtf[28],terms,su.f)
s5=su.gvsm_approx_similarity(tf[5850],qtf[28],terms,su.f)
s6=su.gvsm_approx_similarity(tf[7482],qtf[28],terms,su.f)
s7=su.gvsm_approx_similarity(tf[1],qtf[28],terms,su.f)



al = su.compute_cosine_similarity(tf,qtf)
#aa = wnu.lowest_common_hypernym("corgi","dog")
#print(list(aa.closure(lambda s: s.hypernyms())))
print(s5)
print(s6)
printdiff(s7,al[1][28])
'''
start = time.time()
'''
a = wnu.custom_similarity("bottle","container")
b = wnu.custom_similarity("ball","dance")
c = wnu.custom_similarity("france","catheter")
print(a)
print(b)
print(c)
'''

al = wnu.compute_cosine_similarity(tf,qtf)

zipp = zip(al[:,3], range(0,11430))
a = sorted(zipp,key = lambda t: t[0])
a = list(a)
for i in range(0,15):
    print(a[-i])
wnu.all_term_sim(wnu.custom_similarity,"Intermidiate-data-structure/our_custom_sim.csv")
#wnu.term_sim_compare(lu.load_terms("Test/Dataset/wordsim.csv"),wnu.custom_similarity)
#print(wnu.query_sims(qtf[5],tf,terms,wnu.custom_similarity))
tot = time.time()-start
print("time elapsed  opt : "+str(tot))
#print("total estimated : "+str(tot*93*1000))




#print("Non-opt : "+str(s1))
#print("Opt : "+str(s2))

#s3=gvsm_similarity_Approx2_dq(d1,q1,terms,sim)
#print(s3)

