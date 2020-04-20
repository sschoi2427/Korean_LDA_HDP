# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 18:16:57 2019

https://blog.naver.com/rickman2/221337023577
https://radimrehurek.com/gensim/wiki.html#latent-dirichlet-allocation
http://www.engear.net/wp/topic-modeling-gensimpython/
https://lovit.github.io/nlp/2018/09/27/pyldavis_lda/


@author: admin
"""
import os

DATA_PATH = os.getcwd().replace('\\','/')
RESULT_PATH = DATA_PATH + "/result/"


import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim') #gensim경고문삭제


from konlpy.tag import Komoran
from gensim import corpora, models, similarities
from pprint import pprint
from gensim.models.coherencemodel import CoherenceModel


engin = Komoran()

doc_list = [] #문서를 저장하는 공간
doc_dic = [] # 문서별 단어 파싱을 저장 noun +
#doc_list = ('2018.000123.123', ''안녕', [('1',''SY),('안녕',NNG)])
#doc_dic = [(''','SY'),('안녕','NNG')]


def doc_pos_tokenizer(doc_pos):
    doc_pos_ryu = []
    for item in doc_pos:
        if item[1] in {'NNG', 'NNP', 'SL'} :
            #doc_pos_ryu.append(item[0]+"_"+item[1])
            doc_pos_ryu.append(item[0])
    return doc_pos_ryu



import csv
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == "__main__":
    f = open(RESULT_PATH + "minwon.txt", 'r', encoding="UTF-8")
    row_data = f.read().replace(",", " ").replace("부산",'').replace("부산시",'')
    row_data = row_data.replace("\n", ',').replace(",,",",").replace(",,,",",").replace('\ufeff','').split(',')
    data = csv.reader(row_data)
    
    num_row = 0
    for row in data:
        doc_pos = engin.pos(row[0])
        doc_pos = doc_pos_tokenizer(doc_pos)
        doc_info = (row[0], doc_pos)
        doc_list.append(doc_info)
        doc_dic.append(doc_pos)
        num_row += 1
        print("{}번째 문장 명사 개수 : ".format(num_row) + str(len(doc_pos)))
    print("총 문장 개수 : " + str(len(doc_dic)))
    print("head : " + "< " + row_data[0] + " >")
    print('\n'+"="*75)
    
    #n그램모델 https://wikidocs.net/21692 
    bigram = models.Phrases(doc_dic, min_count=5, threshold=100)
    trigram = models.Phrases(bigram[doc_dic], threshold=100)
    bigram_mod = models.phrases.Phraser(bigram)
    trigram_mod = models.phrases.Phraser(trigram)
    
    
    doc_dic_bi_tri = []
    
    for i in range(0, len(doc_dic)):
        doc_dic_bi_tri.append(trigram_mod[bigram_mod[doc_dic[i]]])
        
    dictionary = corpora.Dictionary(doc_dic_bi_tri)
    dictionary.save('dictionary.dict')
    
    corpus = [dictionary.doc2bow(a_doc_dic_bi_tri) for a_doc_dic_bi_tri in doc_dic_bi_tri]
    corpora.MmCorpus.serialize('corpora.mm', corpus)
    
#    tfidf = models.TfidfModel(corpus)
#    corpus_tfidf = tfidf[corpus]  
    
    print("\n※※※ 한국어텍스트전처리와 n그램모델화가 완료되었습니다. ※※※")
    
###########################################################################################################
    
    
    ml = []
    numl = []
    q0 = str(input("Q. LDA 정확도 계산을 진행하시겠습니까? 시간이 오래걸립니다. (y/n) : "))
    if q0 == 'y' :
        print("\nA. LDA 정확도 계산을 시작합니다. 시간이 오래걸립니다.(10부터 200까지 20단위)")
        print("수치가 낮을수록 정확합니다.\n")
        for num in range(10,150, 20):
            lda_model = models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary,
                                                 num_topics = num,
                                                 random_state=100,
                                                 update_every=1,
                                                 chunksize=2000,
                                                 passes=10,
                                                 eval_every = 10,
                                                 alpha="auto",
                                                 eta=None,
                                                 per_word_topics=True,
                                                 iterations = 1000)
            
            m = lda_model.log_perplexity(corpus)
            print("토픽 " + str(num) + '개 Perplexity : ', m) # https://swenotes.wordpress.com/tag/nlp/
            ml.append(m)
            numl.append(num)

        plt.plot(numl, ml)
        plt.xlabel('K')
        plt.ylabel('Perplexity')
        plt.grid(True)
        plt.show()
            
            
###########################################################################################################

    print("\n※※※ LDA/HDP분석을 시작합니다. ※※※\n")
    topic_num_LDA = int(input("LDA토픽개수 : "))    # LDA 토픽 개수
    topic_num_HDP = int(input("HDP토픽개수 : "))    # HDP 토픽 개수 / -1을 입력 => T_max_num_of_HDP의 값
    word_num = int(input("LDA/HDP단어개수 : "))          #단어 개수
    T_max_num_of_HDP = 200  # HDP 최대 토픽 개수


###########################################################################################################
    
    
    print("\n"+"="*25 + "LDA 모델링" + "="*25)
    
    lda_model = models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary,
                                         num_topics = topic_num_LDA,
                                         random_state=100,
                                         update_every=1,
                                         chunksize=2000,
                                         passes=10,
                                         alpha="auto",
                                         per_word_topics=True,
                                         iterations = 1000)
    print()
    m = lda_model.log_perplexity(corpus)
    print(str(topic_num_LDA)+'개 토픽 Perplexity: ', m)
    
#    def get_topic_term_prob(lda_model):
#        topic_term_freqs = lda_model.state.get_lambda()
#        topic_term_prob = topic_term_freqs / topic_term_freqs.sum(axis=1)[:, None]
#        return topic_term_prob    
#    
#    topic_term_prob = get_topic_term_prob(lda_model)
#    print(topic_term_prob[0].sum())
#    print(topic_term_prob.shape) # (n_topics, n_terms)
        
    topic_num_LDA = int(input("총 {}개 토픽 중 나타낼 토픽 개수 : ".format(topic_num_LDA)))
    word_num = int(input("총 {}개 단어 중 나타낼 단어 개수 : ".format(word_num)))
    
    lda_topic = lda_model.print_topics(num_topics = topic_num_LDA, num_words = word_num)
    df0 = pd.DataFrame(lda_topic)
    df0.columns = ['x', 'result']
    del df0['x']
    print(df0 + "\n")
    xlsx_name = RESULT_PATH + 'LDA_result' + '.xlsx'
    df0.to_excel(xlsx_name, encoding='utf-8')
    
   
    
    print("\n")

###########################################################################################################


    print("="*25 + "HDP 모델링" + "="*25)

    hdp_model = models.hdpmodel.HdpModel(corpus=corpus, id2word=dictionary,
                                        max_chunks=None,
                                        max_time=None,
                                        chunksize=256,
                                        kappa=1.0,
                                        tau=64.0,
                                        K=15,
                                        T = T_max_num_of_HDP,
                                        alpha=1,
                                        gamma=1,
                                        eta=0.01,
                                        scale=1.0,
                                        var_converge=0.0001,
                                        outputdir=None,
                                        random_state=100)
    
    topic_num_HDP = int(input("총 {}개 토픽 중 나타낼 토픽 개수 : ".format(topic_num_HDP)))
    word_num = int(input("총 {}개 단어 중 나타낼 단어 개수 : ".format(word_num)))
    
    hdp_topic = hdp_model.print_topics(num_topics = topic_num_HDP, num_words = word_num)
    df1 = pd.DataFrame(hdp_topic)
    df1.columns = ['x', 'result']
    del df1['x']
    print(df1)
    xlsx_name = RESULT_PATH + 'HDP_result' + '.xlsx'
    df1.to_excel(xlsx_name, encoding='utf-8')
        
    
    
###########################################################################################################    

    

'''
# 시각화 코드 입니다. 위의 코드부터 이 코드까지 복사 한 후 쥬피터에서 실행시켜야합니다.
# pip install pyldavis

import pyLDAvis.gensim
pyLDAvis.enable_notebook()
pyLDAvis.gensim.prepare(lda_model, corpus, dictionary)

'''




# 글자만 추출하는 코드입니다. 수정 절대 하면 안됩니다. ========================
#    for topic in lda_model.show_topics(num_topics = num0, num_words = num1):
#        words = ''
#        for word, prob in lda_model.show_topic(topic[0], topn = num1):
#            words += word + '  '
#        lda_topic = 'Topic{}: '.format(topic[0]) + words
#        print(lda_topic)
#        print()
#        i += 1
#===================================================================================