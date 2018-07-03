import similarity_utilities as su
import load_utilities as lu
import test_functions as test_f
import time
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import numpy as np
import sklearn as sk

#with 50 query -> 70-100 sec for query ==> 1-1.4 h

N_QUERY = 50



start= time.time()
su.sim("house","box")
print("If it is greater than 0.1 secs it will never terminate  "+str(time.time()-start))

tf = lu.load_sparse("Intermidiate-data-structure/tfidf_doc_matrix.npz")
qtf = lu.load_sparse("Intermidiate-data-structure/tfidf_query_matrix.npz")
terms = lu.load_terms("Intermidiate-data-structure/termID_mapping_list.txt")


relevant_doc = [] #indicizzati per query
all_doc = set()

for i in range(N_QUERY):

    relevant_doc.append( lu.load_relevant_for_query(i, "Test/Dataset/rlv-ass.txt"))
    all_doc.update(relevant_doc[i])


all_doc = list(all_doc)
#print(len(all_doc))

N_DOC = len(all_doc)

final_score = np.zeros((N_QUERY,N_DOC,2))

final_score_cos_sim = np.zeros((N_QUERY,N_DOC,2))

#print(all_doc)

print("\nTASK 1 : Computing scores on query, doc : "+str(N_QUERY-1)+" "+str(N_DOC)+"\n")

count=0

for i in range(1,N_QUERY):
    start = time.time()
    query = qtf[i]
    print("\nQuery N: "+str(i)+" on "+str(N_QUERY-1))


    for j in range(N_DOC):
        print(j)
        doc = tf[ all_doc[j] ]
        s = su.gvsm_approx_similarity(doc, query, terms, su.sim)
        #s=sk.metrics.pairwise.cosine_similarity(doc,query)


        final_score[i][j][0] = s
        final_score[i][j][1] = np.int32(all_doc[j])

        #print(str(s)+" "+str(all_doc[j]))
        count+=1
        if (count > 20):
            print("--- Complete : "+str(100*j/N_DOC)+"%")
            #print("Doc N: "+str(j)+" on "+str(N_DOC))
            count = 0
    print("time : "+str(time.time()-start))
    print("--- Complete : " + str(100 ) + "%")
print("\nTASK 2 : Sorting scores\n")


for i in range (1,N_QUERY):
    final_score[i] = final_score[i][final_score[i][:,0].argsort()]



precision = [0]*10 #i-th index is the precision at recall (i+1)*0,1

single_query = [0]*10

file = open("gvsm-report.txt","w")
print("\nTASK 3 : Precision-Recall\n")

for i in range(1,N_QUERY):
    print("\n\n\n\nRelevant docs for the query : "+str(i))
    file.write("\n\n\n\nRelevant docs for the query : "+str(i)+"\n")

    den_recall = len(relevant_doc[i])
    den_precision = 0.0
    num_recall = 0.0
    recall_level = 1
    tot_relevant = 0.0

    for j in reversed(range(N_DOC)):
        den_precision+=1
        #print("score : "+str(final_score[i][j][0]))
        #print("docID : " + str(final_score[i][j][1]))
        if (final_score[i][j][1] in relevant_doc[i]):
            print(" docID : "+str(final_score[i][j][1])+" score : "+str(final_score[i][j][0]))
            file.write(" docID : "+str(final_score[i][j][1])+" score : "+str(final_score[i][j][0])+"\n")
            num_recall+=1
            #print("recall"+str(num_recall/den_recall))
            #print("precision "+str(num_recall/den_precision))
       # print("recall level "+str(recall_level*0.1))
        if (recall_level*0.1<=  num_recall/den_recall and recall_level<11 ):

            #print("RECALL"+str(num_recall/den_recall))
            #print("PRECISION "+str(num_recall/den_precision))
            precision[recall_level-1] += num_recall/den_precision
            single_query[recall_level - 1] = num_recall / den_precision
            recall_level += 1
    print("The single score for the query : " + str(single_query))
    file.write("The single score for the query : " + str(single_query) + "\n")
    #print(precision)

precision = list(map(lambda x: x/(N_QUERY-1), precision))
print("The final score for all the queries : "+str(precision))
file.write("\n\nThe final score for all the queries : "+str(precision)+"\n")
file.close()
#print(su.gvsm_approx_similarity(tf[813], qtf[4], terms, su.f))
#score : 0.31237573126322477
#docID : 813.0
'''
The (score docs) for the query : 2
score : 0.4007771964051852
docID : 1239.0
recall level 0.1

The (score docs) for the query : 1
score : 0.02021351690760601
docID : 4569.0
'''

#print(su.gvsm_approx_similarity(tf[4569], qtf[1], terms, su.sim))


plt.plot([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0] ,precision, '-o')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.1])
plt.xlim([0.0, 1.1])
plt.show()
