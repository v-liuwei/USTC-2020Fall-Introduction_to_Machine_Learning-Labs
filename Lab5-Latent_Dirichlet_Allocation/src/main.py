import os
import re
import nltk
import numpy as np
from string import punctuation
from nltk.corpus import stopwords
from collections import Counter
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from model import LatentDirichletAllocation


def get_vocab_and_doc_word_mat(texts):
    def replace_abbreviations(text):
        text = text.lower()
        # 只保留字母、空格、单引号(所有格及is缩写)
        text = re.sub(r'[^a-z \']+', ' ', text)
        text = re.sub("(it|he|she|that|this|there|here)(\'s)", r"\1 is", text)  # is 缩写
        text = re.sub("(?<=[a-z])\'s", "", text)  # 所有格
        text = re.sub("(?<=s)\'s?", "", text)  # 复数所有格
        text = re.sub("(?<=[a-z])n\'t", " not", text)  # not 缩写
        text = re.sub("(?<=[a-z])\'d", " would", text)  # would 缩写
        text = re.sub("(?<=[a-z])\'ll", " will", text)  # will 缩写
        text = re.sub("(?<=i)\'m", " am", text)  # am 缩写
        text = re.sub("(?<=[a-z])\'re", " are", text)  # are 缩写
        text = re.sub("(?<=[a-z])\'ve", " have", text)  # have 缩写
        text = text.replace('\'', ' ')  # 剩下的单引号去掉
        return text

    lmtzr = WordNetLemmatizer()

    def lemmatize(word):
        def get_wordnet_pos(treebank_tag):
            if treebank_tag.startswith('J'):
                return nltk.corpus.wordnet.ADJ
            elif treebank_tag.startswith('V'):
                return nltk.corpus.wordnet.VERB
            elif treebank_tag.startswith('N'):
                return nltk.corpus.wordnet.NOUN
            elif treebank_tag.startswith('R'):
                return nltk.corpus.wordnet.ADV
            else:
                return ''
        tag = nltk.pos_tag(word_tokenize(word))
        pos = get_wordnet_pos(tag[0][1])
        if pos:
            word = lmtzr.lemmatize(word, pos)
        return word

    blacklist = stopwords.words('english') + ['would', 'may', 'might', 'could', 'shall']
    blacklist.extend(list('bcdefghjklmnopqrstuvwxyz'))
    blacklist.extend(list(punctuation))
    doc_words = []
    vocab = set()
    for text in texts:
        words = replace_abbreviations(text).strip().split()  # 展开缩写
        new_words = []
        for word in words:
            word = lemmatize(word)  # 词性还原
            if word not in blacklist:  # 去除停用词
                new_words.append(word)
                vocab.add(word)
        doc_words.append(new_words)
    vocab = np.array(list(vocab))
    doc_word_mat = np.zeros((len(doc_words), len(vocab)), dtype=np.int)
    for d in range(len(doc_words)):
        counter = Counter(doc_words[d])
        for t in range(len(vocab)):
            doc_word_mat[d][t] = counter[vocab[t]] if vocab[t] in counter else 0
    return vocab, doc_word_mat


if __name__ == '__main__':
    # load and preprocess data, save result
    if not os.path.exists('vocab.npy'):
        docs = np.load('text.npy')
        vocab, doc_word_mat = get_vocab_and_doc_word_mat(docs)
        np.save('vocab.npy', vocab)
        np.save('doc_word_mat.npy', doc_word_mat)
    else:
        vocab = np.load('vocab.npy')
        doc_word_mat = np.load('doc_word_mat.npy')

    # remove low frequent words
    threshold = 20
    selected_words = doc_word_mat.sum(axis=0) > threshold
    vocab = vocab[selected_words]
    doc_word_mat = doc_word_mat[:, selected_words]

    # create a LDA model, fit and transfrom data
    lda = LatentDirichletAllocation(n_topics=20, max_iter=100, random_state=123)
    doc_topic_distr = lda.fit_transform(doc_word_mat)

    # get topic word distribution of fitted model
    topic_word_distr = lda.topic_word_distr

    # list top 10 key words of each topic
    idx = np.argsort(-topic_word_distr, axis=1)
    topic_top10_words = vocab[idx[:, :10]]
    for top10_words in topic_top10_words.tolist():
        print(top10_words)
