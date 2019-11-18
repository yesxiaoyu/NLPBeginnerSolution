#  -*- coding: utf-8  -*-
# best acc score: 0.76526
__author__ = "lihongyu"

import pandas as pd, numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from scipy.sparse import hstack

def sentences_to_bag_of_words(all_sentences):
    vocabSet = []
    for sentence in all_sentences:
        for word in sentence.split():
            if word not in vocabSet:
                vocabSet.append(word)
    return vocabSet

def text_to_vector(bow, sentences):
    res = []
    for sentence in sentences:
        feature = [0] * len(bow)
        for word in sentence.split():
            if word in bow:
                feature[bow.index(word)] = 1
        res.append(feature)
    return res

class ChiSquare:
    def __init__(self, doc_list, doc_labels):
        self.total_data, self.total_pos_data, self.total_neg_data = {}, {}, {}
        for i, doc in enumerate(doc_list):
            if doc_labels[i] == 1:
                for word in doc.split():
                    self.total_pos_data[word] = self.total_pos_data.get(word, 0) + 1
                    self.total_data[word] = self.total_data.get(word, 0) + 1
            else:
                for word in doc.split():
                    self.total_neg_data[word] = self.total_neg_data.get(word, 0) + 1
                    self.total_data[word] = self.total_data.get(word, 0) + 1

        total_freq = sum(self.total_data.values())
        total_pos_freq = sum(self.total_pos_data.values())
        # total_neg_freq = sum(self.total_neg_data.values())

        self.words = {}
        for word, freq in self.total_data.items():
            pos_score = self.__calculate(self.total_pos_data.get(word, 0), freq, total_pos_freq, total_freq)
            # neg_score = self.__calculate(self.total_neg_data.get(word, 0), freq, total_neg_freq, total_freq)
            self.words[word] = pos_score * 2

    @staticmethod
    def __calculate(n_ii, n_ix, n_xi, n_xx):
        n_ii = n_ii
        n_io = n_xi - n_ii
        n_oi = n_ix - n_ii
        n_oo = n_xx - n_ii - n_oi - n_io
        return n_xx * (float((n_ii*n_oo - n_io*n_oi)**2) /
                       ((n_ii + n_io) * (n_ii + n_oi) * (n_io + n_oo) * (n_oi + n_oo)))

    def best_words(self, num, need_score=False):
        words = sorted(self.words.items(), key=lambda word_pair: word_pair[1], reverse=True)
        if need_score:
            return [word for word in words[:num]]
        else:
            return [word[0] for word in words[:num]]

def main():
    train = pd.read_csv('./sentiment-analysis-on-movie-reviews/train.tsv', sep='\t')
    test = pd.read_csv('./sentiment-analysis-on-movie-reviews/test.tsv', sep='\t')

    train_texts = list(train['Phrase'].values)
    train_labels = train['Sentiment'].values
    test_texts = list(test['Phrase'].values)
    x_train, x_test, y_train, y_test = train_test_split(train_texts, train_labels, test_size=0.2)

    all_text = train_texts + test_texts
    print('all_text shape: {} & all_text0: {}'.format(len(all_text), all_text[0]))

    bow = sentences_to_bag_of_words(all_text)
    print('bow list longth', len(bow))  # 21637
    k = 1000

    fe = ChiSquare(train_texts, train_labels)
    best_words = fe.best_words(k)
    print(len(best_words))

    train_bow_feature = np.array(text_to_vector(best_words, x_train))
    test_bow_feature = np.array(text_to_vector(best_words, x_test))
    print(train_bow_feature.shape, test_bow_feature.shape)

    # 提取文本计数特征 -- 每个单词的数量
    # 对文本的单词进行计数，包括文本的预处理, 分词以及过滤停用词
    count_vect = CountVectorizer()
    x_train_counts = count_vect.fit_transform(x_train)
    x_test_counts = count_vect.transform(x_test)
    print(x_train_counts.shape, x_test_counts.shape)  # (93636, 15188) (31212, 15188)  矩阵(句子-词汇）的维度，词表大小15188
    # 在词汇表中一个单词的索引值对应的是该单词在整个训练的文集中出现的频率。
    # print(count_vect.vocabulary_.get(u'good'))    #5812     count_vect.vocabulary_是一个词典：word-id
    
    # 提取TF-IDF特征 -- word级别的TF-IDF
    # 将各文档中每个单词的出现次数除以该文档中所有单词的总数：这些新的特征称之为词频tf。
    tfidf_transformer = TfidfVectorizer(analyzer='word', max_features=50000)
    tfidf_transformer.fit(x_train)
    x_train_tfidf_word = tfidf_transformer.transform(x_train)
    x_test_tfidf_word = tfidf_transformer.transform(x_test)
    print(x_train_tfidf_word.shape, x_test_tfidf_word.shape)

    # 提取TF-IDF特征 - ngram级别的TF-IDF
    # 将各文档中每个单词的出现次数除以该文档中所有单词的总数：这些新的特征称之为词频tf。
    tfidf_transformer = TfidfVectorizer(analyzer='word', ngram_range=(2, 3), max_features=50000)
    tfidf_transformer.fit(x_train)
    x_train_tfidf_ngram = tfidf_transformer.transform(x_train)
    x_test_tfidf_ngram = tfidf_transformer.transform(x_test)
    print(x_train_tfidf_ngram.shape, x_test_tfidf_ngram.shape)

    # 合并特征（特征组合与特征选择）
    
    train_features = hstack([train_bow_feature, x_train_counts, x_train_tfidf_word, x_train_tfidf_ngram])
    test_features = hstack([test_bow_feature, x_test_counts, x_test_tfidf_word, x_test_tfidf_ngram])

    print(train_features.shape)  # 特征的最终维度
    print(test_features.shape)

    train_features, y_train = shuffle(train_features, y_train)

    clf = LogisticRegression(random_state=2019, solver='saga',  # 优化算法：liblinear、lbfgs、newton-cg、sag
                             multi_class='multinomial',  # 分类方式：multinomial、ovr
                             max_iter=1000).fit(train_features, y_train)
    # 使用LR，只有bag-of-word特征的结果为：0.6023644752018454
    # 使用LR，只有词频特征的结果为：0.6497180571575035
    # 使用LR，包括词频、TFIDF、3-gram-TFIDF特征的结果为：0.6671472510572857
    # 使用LR，包括BOW、词频、TFIDF、3-gram-TFIDF特征的结果为：0.6713123157759836

    # clf = MultinomialNB().fit(train_features, y_train) # 0.5913110342176087
    # clf = SGDClassifier(alpha=0.001,
    #                     loss='hinge',  # hinge代表SVM，log是逻辑回归
    #                     early_stopping=True,
    #                     eta0=0.001,
    #                     learning_rate='adaptive',  # constant、optimal、invscaling、adaptive
    #                     max_iter=100
    #                     ).fit(train_features, y_train) # log：0.5452390106369345 hinge:0.5552992438805587
    predict = clf.predict(test_features)
    print(np.mean(predict == y_test))

if __name__ == '__main__':
    main()