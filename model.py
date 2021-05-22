from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from gensim import corpora
import gensim
import numpy as np
from Autoencoder import *
from preprocess import *
from datetime import datetime

def preprocess(docs, sample_size=None):
    if not sample_size:
        sample_size = 100

    n_docs = len(docs)
    samp = np.random.choice(n_docs, sample_size)
    sentences = []
    index_of_selected_sample = []
    list_of_tokens = []
    print('Preprocessing raw texts ...')

    for i, idx in enumerate(samp):
        sentence = preprocess_sentence(docs[idx])
        list_of_tokens = preprocess_word(sentence)
        if list_of_tokens:
            index_of_selected_sample.append(idx)
            sentences.append(sentence)
            list_of_tokens.append(list_of_tokens)
        print('{} %'.format(str(np.round((i + 1) / len(samp) * 100, 2))), end='\r')
    print('Preprocessing raw texts done')
    return sentences, list_of_tokens, index_of_selected_sample

class Topic_Model:
    def __init__(self, k=10, method='TFIDF'):
        if method not in {'TFIDF', 'LDA', 'BERT', 'LDA_BERT'}:
            raise Exception('Invalid method!')
        self.k = k
        self.dictionary = None
        self.corpus = None
        #         self.stopwords = None
        self.cluster_model = None
        self.ldamodel = None
        self.vec = {}
        self.gamma = 15
        self.method = method
        self.AE = None
        self.id = method + '_' + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self.num_topics = 10

    def vectorize(self, sentences, list_of_tokens, method=None):
        if method is None:
            method = self.method

        self.dictionary = corpora.Dictionary(list_of_tokens)
        self.corpus = [self.dictionary.doc2bow(text) for text in list_of_tokens]

        if method == 'TFIDF':
            print('Vector representations for TF-IDF:')
            tfidf = TfidfVectorizer()
            vec = tfidf.fit_transform(sentences)
            print('Vector representations for TF-IDF done')
            return vec

        elif method == 'LDA':
            print('Vector representations for LDA:')
            if not self.ldamodel:
                self.ldamodel = gensim.models.ldamodel.LdaModel(self.corpus, num_topics=self.k, id2word=self.dictionary,
                                                                passes=20)

            def get_lda_vectorised_representation(model, corpus, k):
                n_doc = len(corpus)
                vectorised_lda = np.zeros((n_doc, k))
                for i in range(n_doc):
                    for topic, prob in model.get_document_topics(corpus[i]):
                        vectorised_lda[i, topic] = prob

                return vectorised_lda

            vec = get_lda_vectorised_representation(self.ldamodel, self.corpus, self.k)
            print('Vector representations for LDA done!')
            return vec

        elif method == 'BERT':
            print('Vector representations for BERT ...')
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('bert-base-nli-max-tokens')
            vec = np.array(model.encode(sentences, show_progress_bar=True))
            print('Vector representations for BERT done')
            return vec

        else:
            vectorised_lda = self.vectorize(sentences, list_of_tokens, method='LDA')
            vectorised_bert = self.vectorize(sentences, list_of_tokens, method='BERT')
            vectorised_ldabert = np.c_[vectorised_lda * self.gamma, vectorised_bert]
            self.vec['LDA_BERT_FULL'] = vectorised_ldabert
            if not self.AE:
                self.AE = Autoencoder()
                print('Fitting Autoencoder ...')
                self.AE.fit(vectorised_ldabert)
                print('Fitting Autoencoder done!')
            vec = self.AE.encoder.predict(vectorised_ldabert)
            return vec

    def fit(self, sentences, list_of_tokens, method=None, m_clustering=None):
        if method is None:
            method = self.method
        if m_clustering is None:
            m_clustering = KMeans

        if not self.dictionary:
            self.dictionary = corpora.Dictionary(list_of_tokens)
            self.corpus = [self.dictionary.doc2bow(text) for text in list_of_tokens]

        if method == 'LDA':
            if not self.ldamodel:
                print('Fitting LDA ...')
                self.ldamodel = gensim.models.ldamodel.LdaModel(self.corpus, num_topics=self.k, id2word=self.dictionary,
                                                                passes=20)
                print('Fitting LDA Done!')
        else:
            print('Clustering embeddings ...')
            self.cluster_model = m_clustering(self.k)
            self.vec[method] = self.vectorize(sentences, list_of_tokens, method)
            self.cluster_model.fit(self.vec[method])
            print('Clustering embeddings done')

    def predict(self, sentences, list_of_tokens, out_of_sample=None):
        out_of_sample = out_of_sample is not None

        if out_of_sample:
            corpus = [self.dictionary.doc2bow(text) for text in list_of_tokens]
            if self.method != 'LDA':
                vec = self.vectorize(sentences, list_of_tokens)
                print(vec)
        else:
            corpus = self.corpus
            vec = self.vec.get(self.method, None)

        if self.method == "LDA":
            lbs = np.array(list(map(lambda x: sorted(self.ldamodel.get_document_topics(x),
                                                     key=lambda x: x[1], reverse=True)[0][0],
                                    corpus)))
        else:
            lbs = self.cluster_model.predict(vec)
        return lbs
