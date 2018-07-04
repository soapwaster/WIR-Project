from nltk.tokenize import word_tokenize as wt
from nltk.corpus import wordnet as wn
import numpy as np
from textblob import TextBlob
import io
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import scipy
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.sparse import csr_matrix, find
from nltk.corpus import wordnet
import load_utilities as lu
import io

import numpy as np
import scipy
from nltk.corpus import wordnet
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize as wt
from scipy.sparse import csr_matrix, find
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob

def no_opt_gvsm_similarity_Approx1_qq_dd_dq(doc, query, term, similarity):
    '''
    TODO: optimization
    Accessing to the tfidf score in the sparse vector doc or query is expensive .
    The optimization can be done becuase find returns the tfidf score associated to a doc or query

    '''

    """
    Returns the approximated similarity score iterating over the union of terms in doc and query
    :param doc: tdidf document vector
    :param query: tdidf query vector
    :param term: array of mapping term
    :param similarity: similarity function
    :return: gvsm score
    """

    row_d,col_d,val_d = find(doc)
    row_q, col_q,val_q = find(query)


    tot = merge(col_d,col_q)


    dim = len(tot)
    score = 0;
    den_doc = 0;
    den_query = 0;




    for i in range(dim):
        d_i = doc[0,tot[i]] #very expensive
        q_i = query[0,tot[i]] #very expensive
        for j in range(i,dim):
            sim = similarity(term[tot[i]],term[tot[j]])
            docij = ( d_i + doc[0,tot[j]] )*sim #very expensive
            queryij =( q_i + query[0,tot[j]] )*sim #very expensive
            score += ( docij  ) * ( queryij )
            den_doc += np.square(docij)
            den_query += np.square(queryij)

    norm = np.sqrt(den_doc*den_query)

    return score/norm

def gvsm_similarity_complete_slow(doc, query, term, similarity):
    """
    Returns the correct similarity score but it is too expensive
    :param doc: tdidf document vector
    :param query: tdidf query vector
    :param term: array of mapping term
    :param similarity: similarity function
    :return: gvsm score
    """
    #row_d,col_d,val_d = find(doc)
    #row_q, col_q, val_q = find(query)
    #print(col_d)
    #print(col_q)



    dim = len(term)
    score = 0;
    den_doc = 0;
    den_query = 0;




    for i in range(dim):
        d_i = doc[0,i]
        q_i = query[0,i]
        for j in range(i,dim):
            sim = similarity(term[i],term[j])
            docij = ( d_i + doc[0,j] )*sim
            queryij =( q_i + query[0,j] )*sim
            score += ( docij  ) * ( queryij )
            den_doc += np.square(docij)
            den_query += np.square(queryij)

    norm = np.sqrt(den_doc*den_query)
    return score/norm

def nounify(adjective):
    """
     :param adjective: the adjective
    :return: set of nouns semantically related to it
    """
    set_of_related_nouns = set()

    for lemma in wn.lemmas(wn.morphy(adjective.lower(), pos="a")):
        if len(lemma.derivationally_related_forms()) == 0:
            for related_form in wn.synsets(lemma.name()):
                for synset in related_form.attributes():
                    set_of_related_nouns.add(synset)
        else:
            for related_form in lemma.derivationally_related_forms():
                for synset in wn.synsets(related_form.name(), pos=wn.NOUN):
                    set_of_related_nouns.add(synset)

    return set_of_related_nouns

def extract_noun_phrases(txt):
    """
    Extract only noun phrases from a sentence
    :param txt: the sentence
    """
    blob = TextBlob(txt)
    print(blob.noun_phrases)

def compute_cosine_similarity(tfidfdocs, tfidfqueries):
    return cosine_similarity(tfidfdocs, tfidfqueries)

def compute_cosine_similarity_from_file(tf, qtf):
    dd = load_documents(tf)
    qq = load_documents(qtf)

    a,b,c = tf_idf(dd,qq)
    return compute_cosine_similarity(a,b)

def lowest_common_hypernym(term1, term2):
    synsets_t1 = wn.synsets(term1,pos="n")
    synsets_t2 = wn.synsets(term2,pos="n")
    max_depth = 0
    lch = ""
    t1 = ""
    t2 = ""

    for s in synsets_t1:
        if s.pos() == 'v':
            continue
        elif s.pos() == 'a':
            synsets_t1.extend(list(nounify(s.lemmas()[0].name())))
            continue
        for t in synsets_t2:
            if t.pos() == 'v':
                continue
            elif t.pos() == 'a':
                synsets_t2.extend(list(nounify(t.lemmas()[0].name())))
                continue

            lc = s.lowest_common_hypernyms(t)
            depth = 0
            if lc != []:
                depth = lc[0].min_depth()
            if lc != [] and  depth > max_depth:
                max_depth = depth
                t1 = s
                t2 = t
                lch = lc[0]
                #lch.append(s.lowest_common_hypernyms(t))

    return lch, max_depth, t1, t2

def custom_similarity(term1, term2):
    lca,lca_depth,terma,termb = lowest_common_hypernym(term1,term2)
    if(lca_depth == 0): return 0
    hop_diff = (terma.min_depth() - lca_depth) + (termb.min_depth() - lca_depth) + 1
    if hop_diff == 0 : hop_diff = 1
    depth_info = np.log(lca_depth) #Devi usare qualche info sulla depth dell'lca
    return (1/(hop_diff**(1/3))) * depth_info

def gvsm_approx_similarity(doc, query, term, similarity):

    """
    Returns the approximated similarity score iterating over the union of terms in doc and query
    :param doc: tdidf document vector
    :param query: tdidf query vector
    :param term: array of mapping term
    :param similarity: similarity function
    :return: gvsm score
    """

    row_d,col_d,val_d = find(doc)
    row_q, col_q,val_q = find(query)







    tot = super_merge(col_d,col_q,val_d,val_q)


    #tot is an array of triples (term_id , tfidf_termID in doc , tfidf_termID in query) sorted by term_id


    dim = len(tot)
    score = 0;
    den_doc = 0;
    den_query = 0;


    i=0
    j=0

    flag1 = False
    flag2 = False
    flag3 = False
    z = 0

    while(i<dim):
        termID_ith = tot[i][0]



        d_i =  tot[i][1]
        q_i = tot[i][2]
        #print(term[termID_ith])
        #print(" d_i q_i " + " " + str(d_i) + " " + str(d_i))
        while (j+i<dim):


            #print(i)
            #print(j)
            termID_jth = tot[j+i][0]

            sim = similarity(term[termID_ith],term[termID_jth])
            docij = ( d_i + tot[j+i][1] )*sim
            queryij =( q_i + tot[j+i][2]  )*sim




            score += ( docij  ) * ( queryij )



            #print(" docij : " + str(docij) + " queryij : " + str(queryij) + " product : " + str((docij) * (queryij)))
            #print("score "+str(score))
            den_doc += np.square(docij)
            den_query += np.square(queryij)

            if (flag2):
                j+=1
                flag2= False
                #if ( not flag3 ): flag3 = False
                continue
            if (flag1):
                j+=1
                flag3 = True
                flag1= False
                continue

            if (tot[j+i][1]==0 or tot[j+i][2]==0 ):
                j+=1

            else :
                flag2 = True
        j=0
        if (flag3):
            flag3=False

            i+=1
            continue



        if (d_i == 0 or q_i == 0):
            i+=1
        else :
            flag1 = True


    norm = np.sqrt(den_doc*den_query)

    return score/norm

def super_merge(a,b,val_d,val_q):
    '''

    :param a: array of integer : termid_doc
    :param b: array of integer : termid_query
    :param val_d: array of integer : val_doc
    :param val_q: array of integer : val_query
    :return: merged sorted array  by termid, of triples (termid,val_doc,val_query)
    '''


    i1=0
    i2=0
    la = len(a)
    lb = len(b)
    lab = la+lb
    ris = []
    for t in range(lab):
        if (i1+i2>lab-1): break;
        if (i1>la-1):
            ris.append( (b[i2],0,val_q[i2]))
            i2+=1
        elif (i2>lb-1) :
            ris.append((a[i1],val_d[i1],0))
            i1+=1

        elif (a[i1]<b[i2]):
            ris.append((a[i1],val_d[i1],0))
            i1+=1;

        elif (b[i2]<a[i1]):
            ris.append((b[i2],0,val_q[i2]))
            i2+=1;


        else :

            ris.append((b[i2],val_d[i1],val_q[i2]))
            i2+=1
            i1+=1


    return ris

def merge(a,b):

    i1=0
    i2=0
    la = len(a)
    lb = len(b)
    lab = la+lb
    ris = []
    for t in range(lab):
        if (i1+i2>lab-1): break;
        if (i1>la-1):
            ris.append( b[i2])
            i2+=1
        elif (i2>lb-1) :
            ris.append(a[i1])
            i1+=1

        elif (a[i1]<b[i2]):
            ris.append(a[i1])
            i1+=1;

        else :
            ris.append(b[i2])
            i2+=1;


    return ris

def f(a,b):
    return 1

def sim(word1,word2):

    s=0
    max = 0
    l1=wordnet.synsets(word1, pos="n")
    l2=wordnet.synsets(word2, pos="n")
    if (l1 and l2):
        for a in l1:
            for t in l2:
                s = a.path_similarity(t)
                if s > max: max = s
    if (s is None):
        max=0
    return max

def query_relevant_top_100(q,docs,terms,sim,list):
    """
    :param q: tfidf vector for the query
    :param docs: tfidf matrix for documents
    :param terms: term vector
    :param sim: similarity function
    :param list: list of documents to compare with
    :return: top 100 documents retrieved by sim
    """
    val = []
    for u,i in enumerate(list):
        #if u == 100: break
        val.append(tuple((gvsm_approx_similarity(docs[i,:],q,terms,sim),i)))
    a = sorted(val, key=lambda t: t[0])
    b = []
    for i in range(1,99):
        b.append(a[-i][1])
    return b

def query_intersection(query_top_100_from_sim, query_real_rel):
    """
    :param query_top_100_from_sim: list of elements found relevant from the custom similarity function
    :param query_real_rel: real relevant documents
    :return: list of elemnts where element i is 1 if the document is relevant
    """

    query_top_100_from_sim = list(map(int, query_top_100_from_sim))
    intersection = list(set(query_real_rel).intersection(query_top_100_from_sim))

    rel_list = []
    for i,el in enumerate(query_top_100_from_sim):
        if(i == len(query_real_rel)): break;
        if el in intersection:
            rel_list.append(1)
        else:
            rel_list.append(-1)
    return rel_list

def term_sim_compare(term, sim):
    """
    :param term: dictionary of terms (t1,t2,v) : v
    :param sim: similarity function to use
    :return pc: pearson coefficient
    """
    a =[]
    b =[]
    for el in term:
        terms = el.split(";")
        similarity = sim(terms[0],terms[1])
        a.append(float(terms[2]))
        b.append(similarity)
        #print(str(terms) + " .... "+ str(similarity))
    return spearmanr(a,b)

def all_term_sim(sim,sim_name):
    """
    Computes all similarity for all the term-term files, and saves them into a file
    :param sim: similarity function
    :param sim_name: name of the file (with extension)
    """
    rg = term_sim_compare(lu.load_terms("Test/Dataset/rg.csv"),sim)
    ws = term_sim_compare(lu.load_terms("Test/Dataset/wordsim.csv"),sim)
    mc = term_sim_compare(lu.load_terms("Test/Dataset/mc.csv"),sim)
    with open(sim_name, 'w') as file:
        file.write('rg;' + str(rg[0])+"\n")
        file.write('ws;' + str(ws[0])+"\n")
        file.write('mc;' + str(mc[0]))

def top_relevant_docs_for_query_from_cosine(tf,qtf,filename):
    """
    Saves into a file the top 100 documents for a given query according to cosine similarity
    :param tf: documents tfidf matrix
    :param qtf: query tfidf matrix
    :param filename: filename to save top relevant docs
    """
    al = compute_cosine_similarity(tf, qtf)
    file = open(filename, 'w')
    for i in range(1,94):
        zipp = zip(al[:, i], range(0, 11430))
        a = sorted(zipp, key=lambda t: t[0])
        a = list(a)
        file.write(str(i) + "\n")
        for j in range(1, 100):
            file.write(str(a[-j][1])+" ")
        file.write("/ \n")
    file.close()


def nltk_similarity(word1,word2):



    wordFromList1 = wordnet.synsets(word1)
    wordFromList2 = wordnet.synsets(word2)
    if wordFromList1 and wordFromList2:  # Thanks to @alexis' note
        s = wordFromList1[0].wup_similarity(wordFromList2[0])
        return s
    else: return 0
