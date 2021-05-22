from collections import Counter
import pyLDAvis as pyLDAvis
from sklearn.metrics import silhouette_score
import umap
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from gensim.models.coherencemodel import CoherenceModel
import numpy as np
import os
from main import find_review
from main import get_rws
from preprocess import preprocess_sentence

def get_topic_words(token_lists, labels, k=None):
    if k is None:
        k = len(np.unique(labels))
    topics = ['' for _ in range(k)]
    for i, c in enumerate(token_lists):
        topics[labels[i]] += (' ' + ' '.join(c))
    word_counts = list(map(lambda x: Counter(x.split()).items(), topics))
    # get sorted word counts
    word_counts = list(map(lambda x: sorted(x, key=lambda x: x[1], reverse=True), word_counts))
    # get topics
    topics = list(map(lambda x: list(map(lambda x: x[0], x[:10])), word_counts))

    return topics

def get_coherence(model, token_lists, measure='c_v'):
    if model.method == 'LDA':
        cm = CoherenceModel(model=model.ldamodel, texts=token_lists, corpus=model.corpus, dictionary=model.dictionary,
                            coherence=measure)
    else:
        topics = get_topic_words(token_lists, model.cluster_model.labels_)
        cm = CoherenceModel(topics=topics, texts=token_lists, corpus=model.corpus, dictionary=model.dictionary,
                            coherence=measure)
    return cm.get_coherence()

def get_silhouette(model):
    if model.method == 'LDA':
        return
    lbs = model.cluster_model.labels_
    vec = model.vec[model.method]
    return silhouette_score(vec, lbs)

def plot_proj(embedding, lbs, method):
    n = len(embedding)
    counter = Counter(lbs)
    plt.title( str(method) +' UMAP')
    for i in range(len(np.unique(lbs))):
        plt.plot(embedding[:, 0][lbs == i], embedding[:, 1][lbs == i], '.', alpha=0.5,
                 label='cluster {}: {:.3f}%'.format(i, counter[i] / n * 100))
    plt.legend()

def visualize(model):
    if model.method == 'LDA':
        return
    reducer = umap.UMAP()
    print('Calculating UMAP projection ...')
    vec_umap = reducer.fit_transform(model.vec[model.method])
    print('Calculating UMAP projection done')
    plot_proj(vec_umap, model.cluster_model.labels_, model.method)
    dr = r'C:\Users\Yulia\Downloads\contextual_topic_identification\docs\images\{}\{}'.format(model.method, model.id)
    if not os.path.exists(dr):
        os.makedirs(dr)
    plt.savefig(dr + '/2D_vis')

def get_wordcloud(model, token_lists, topic):
    print('Getting wordcloud for topic {} ...'.format(topic))
    lbs = model.cluster_model.labels_
    tokens = ' '.join([' '.join(_) for _ in np.array(token_lists)[lbs == topic]])
    dr = r'C:\Users\Yulia\Downloads\contextual_topic_identification\docs\images\{}\{}'.format(model.method, model.id)
    if not os.path.exists(dr):
        os.makedirs(dr)
    chunks = tokens.split(' ')
    count = Counter(chunks).most_common()
    word = []
    occurrence = []

    for each in count:
        word.append(each[0])
        occurrence.append(each[1])

    word_of_interest = []
    occurrence_of_interest = np.array([])
    WORDS_OF_INTEREST = ['bug', 'problem', 'server', 'cheat', 'hack']  # WORDS_OF_INTEREST
    rws = get_rws()
    sentences_for_check = []
    for i in range(len(rws)):
        sentence = preprocess_sentence(rws[i])
        sentences_for_check.append(sentence)
    topic_sentences_to_check = find_review(word, sentences_for_check, rws, True)
    print('topic_sentences_to_check ' + str(len(topic_sentences_to_check[0])))
    found_reviews = find_review(WORDS_OF_INTEREST, topic_sentences_to_check[0], topic_sentences_to_check[1])
    file_reviews = open(dr + '/Topic' + str(topic) + "example_reviews"
                                                     ".txt", "w",
                 encoding='utf-8')
    file_reviews.write(str(found_reviews))
    file_reviews.close()

    sum_occurrence_of_interest = 0
    for i in range(0, len(word)):
        if check_word_of_interest(word[i], WORDS_OF_INTEREST) == 0:
            word_of_interest.append(word[i])
            occurrence_of_interest = np.append(occurrence_of_interest, occurrence[i])
            sum_occurrence_of_interest += occurrence[i]

    explode = np.zeros(len(word_of_interest))
    if len(explode) > 0:
        explode[0] = 0.1
    explode = tuple(explode)
    fig1, ax1 = plt.subplots()
    ax1.pie(occurrence_of_interest, explode=explode, labels=word_of_interest, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    plt.title('Topic ' + str(topic) + ': words of interest pie chart')
    plt.savefig(dr + '/Topic' + str(topic) + '_words_of_interest-pie_chart')

    sum_occurrence = 0
    for i in range(0, len(word)):
        sum_occurrence += occurrence[i]

    chart_data = [sum_occurrence_of_interest, sum_occurrence - sum_occurrence_of_interest]
    chart_data_labels = ['words of interest', 'rest']

    explode2 = np.zeros(len(chart_data_labels))
    if len(explode2) > 0:
        explode2[0] = 0.1
    explode2 = tuple(explode2)
    fig1, ax1 = plt.subplots()
    ax1.pie(chart_data, explode=explode2, labels=chart_data_labels, autopct='%1.1f%%',
            shadow=True, startangle=0)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    plt.title('Topic ' + str(topic) + ': words of interest overall percentage pie chart')
    plt.savefig(dr + '/Topic' + str(topic) + '_words_of_interest-pie_chart-overall')

    plt.rcdefaults()
    fig, ax = plt.subplots()
    y_pos = np.arange(len(word[:20]))
    ax.barh(y_pos, occurrence[:20], align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(word[:20])
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Prevalence')
    ax.set_title('Topic ' + str(topic) + ': prevalence bar plot')
    plt.savefig(dr + '/Topic' + str(topic) + '_bar')
    wordcloud = WordCloud(width=800, height=560,
                          background_color='white', collocations=False,
                          min_font_size=10).generate(tokens)
    plt.figure(figsize=(8, 5.6), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)

    plt.savefig(dr + '/Topic' + str(topic) + '_wordcloud')
    print('Topic {} wordcloud done'.format(topic))

def check_word_of_interest(word, words_of_interest):
    for word_of_interest in words_of_interest:
        if word_of_interest in word:
            return 0
    return -1