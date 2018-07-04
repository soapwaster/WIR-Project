import similarity_utilities as wnu
import load_utilities as lu
import numpy as np

def test_query(query_id,tf,qtf,terms):
    relevant_docs = lu.load_all_relevant_docs("Test/Dataset/rlv-ass.txt")
    query_rel_docs = wnu.query_relevant_top_100(qtf[query_id], tf, terms, wnu.custom_similarity, relevant_docs)
    real_rel = lu.load_relevant_for_query(query_id,"Test/Dataset/rlv-ass.txt")
    inters = wnu.query_intersection(query_rel_docs, real_rel)
    thefile = open('relevance.txt', 'a')
    thefile.write(query_id)
    for el in inters:
        thefile.write(el)
    thefile.write("/")
    return inters

def precision_recall(vals):
    vals = vals.strip().split(",")
    print(vals)
    no_r = vals.count(' 1')
    tp = 1
    c = 1
    p = [1]
    r = [0]
    for el in vals:
        if el == ' 1' :
            tp += 1
        c +=1
        p.append(tp/c)
        r.append(tp/no_r)
    return p,r
def precision_recall_cosine_temp(vals):
    no_r = vals.count(1) +1
    tp = 1
    c = 1
    p = [1]
    r = [0]
    for el in vals:
        if el == 1:
            tp += 1
        c += 1
        p.append(tp / c)
        r.append(tp / no_r)
    return p, r

def interpolated_precision_recall(list_of_query_id):
    pr_f = [0,0,0,0,0,0,0,0,0,0,0]
    pr_r = [0,0,0,0,0,0,0,0,0,0,0]
    for i in list_of_query_id:
        v = lu.load_documents("Test/Relevance/relevance_"+str(i)+".txt")
        p,r = precision_recall(v[1])
        for j,el in enumerate([0,0.1,.2,.3,.4,.5,.6,.7,.8,.9,1]):
            pr_f[j] += sum_range(el,el+0.1,r,p)
            if sum_range(el,el+.1,r,p) > 0:
                pr_r[j]+=1
    for i,el in enumerate(pr_f):
        pr_f[i] = el/pr_r[i]
    return pr_f

def interpolated_precision_recall_cos(list_of_query_id,v):
    pr_f = [0,0,0,0,0,0,0,0,0,0,0]
    pr_r = [0,0,0,0,0,0,0,0,0,0,0]
    for i in list_of_query_id:
        if i < 50 :
            p,r = precision_recall_cosine_temp(v[i])
            for j,el in enumerate([0,0.1,.2,.3,.4,.5,.6,.7,.8,.9,1]):
                pr_f[j] += sum_range(el,el+0.1,r,p)
                if sum_range(el,el+.1,r,p) > 0:
                    pr_r[j]+=1
    for i,el in enumerate(pr_f):
        pr_f[i] = el/pr_r[i]
    return pr_f

def sum_range(start,end,list,list2):
    p = 0
    c = 1
    for j,i in enumerate(list):
        if i >= start and i <end:
            return list2[j]
            c=c+1
    return p/c