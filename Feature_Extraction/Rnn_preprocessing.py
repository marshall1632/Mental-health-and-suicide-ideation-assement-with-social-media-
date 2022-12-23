import csv
import itertools
import sys

import nltk
import numpy as np


def getSentenceData(path, vocabulary_size=8000):
    unknown_token = "UNKNOWN_TOKEN"
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, skipinitialspace=True)
        sentences = itertools.chain(*[nltk.sent_tokenize(x[0].lower()) for x in reader])
        sentences = ["%s" % x for x in sentences]
    print("Parsed %d sentences." % (len(sentences)))
    tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]
    tokenized_sentences = list(filter(lambda x: len(x) > 3, tokenized_sentences))

    word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))

    vocab = word_freq.most_common(vocabulary_size - 1)
    index_to_word = [x[0] for x in vocab]
    index_to_word.append(unknown_token)
    word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])

    for i, sent in enumerate(tokenized_sentences):
        tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]

    X_trains = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences], dtype=object)
    y_trains = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences], dtype=object)

    return X_trains, y_trains


def loadGloveModel(emb_path):
    print("Loading Glove Model")
    File = emb_path
    f = open(File, encoding='utf8')
    gloveModel = {}
    for lin in f:
        splitLines = lin.split()
        word = splitLines[0]
        wordEmbedding = np.array([float(value) for value in splitLines[1:]])
        gloveModel[word] = wordEmbedding
    print(len(gloveModel), " words loaded!")
    return gloveModel