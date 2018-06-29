import similarity_utilities as wnu
import load_utilities as lu
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
    no_r = vals.count(1)
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