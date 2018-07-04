import similarity_utilities as wnu
import load_utilities as lu
import test_functions as test_f
import time
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import numpy as np
from matplotlib.legend_handler import HandlerLine2D

#d = lu.load_documents("Test/Dataset/document.txt")
#q = lu.load_documents("Test/Dataset/query.txt")
#a,b,term = lu.tf_idf(d,q)
'''
tf = lu.load_sparse("Intermidiate-data-structure/tfidf_doc_matrix.npz")
qtf = lu.load_sparse("Intermidiate-data-structure/tfidf_query_matrix.npz")
terms = lu.load_terms("Intermidiate-data-structure/termID_mapping_list.txt")
'''
#al = su.compute_cosine_similarity(tf,qtf)

#wnu.all_term_sim(wnu.custom_similarity,"Intermidiate-data-structure/our_custom_sim.csv")
#wnu.term_sim_compare(lu.load_terms("Test/Dataset/wordsim.csv"),wnu.custom_similarity)
#common = test_f.test_query(27,tf,qtf,terms)
#print(common)
'''
val = lu.load_documents("Test/Relevance/relevance_37.txt")
ree = lu.load_all_relevant_docs("Test/Dataset/rlv-ass.txt")
q_id=3
#real_rell = lu.load_relevant_for_query(q_id,"Test/Dataset/rlv-ass.txt")
#v = wnu.query_relevant_top_100(qtf[q_id],tf,terms,wnu.custom_similarity,ree)
#aa = wnu.query_intersection(v,real_rell)
#lu.save_query_intersection(q_id,aa)


qid = 37

val2 = lu.load_documents("Intermidiate-data-structure/rlv-cosine.txt")
val2[qid] = val2[qid].strip().split(" ")
real_rel = lu.load_relevant_for_query(qid, "Test/Dataset/rlv-ass.txt")
vv = wnu.query_intersection(val2[qid],real_rel)
print(vv)
precision, recall = test_f.precision_recall_cosine_temp(vv)
print(precision)
print(recall)

plt.plot(recall, precision, '')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.0])
plt.xlim([0.0, 1.0])
plt.show()

precision, recall = test_f.precision_recall(val[1])
print(precision)
print(recall)

plt.plot(recall, precision, '')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.0])
plt.xlim([0.0, 1.0])
plt.show()

'''
val2 = lu.load_documents("Intermidiate-data-structure/rlv-cosine.txt")
vv = []
vv.append(0)
for i in range(1,51):
    zz = val2[i].strip().split(" ")
    real_rel = lu.load_relevant_for_query(i, "Test/Dataset/rlv-ass.txt")
    vv.append(wnu.query_intersection(zz,real_rel))

final_val = test_f.interpolated_precision_recall_cos(list(range(1,51)),vv[1:51])


valori = test_f.interpolated_precision_recall([3,7,14,22,25,26,37,41,46,56,61,68,75,93])
a = plt.plot([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1], valori, '', marker='o', label="GVSM")
b = plt.plot([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1], final_val, '', marker='o', label="VSM")

plt.gca().legend(('GVSM','VSM'))
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.0])
plt.xlim([0.0, 1.0])
plt.show()
#wnu.all_term_sim(wnu.custom_similarity,"Intermidiate-data-structure/Spearman-colittas-sim2.csv")