import sys
import numpy as np

from Feature_Extraction.Rnn_preprocessing import loadGloveModel, getSentenceData
from Models.RNN.rnn_model import RNNNumpy

glove_file = 'glove.6B.300d.txt'
emb_matrix = loadGloveModel(glove_file)

word_dim = emb_matrix['memory'].shape[0]
hidden_dim = 64
X_train, Y_train = getSentenceData('tweets_train.txt', word_dim)
# X_test, Y_test = preProcess.getSentenceData('tweets_test.txt', word_dim)
np.random.seed(10)

rnn = RNNNumpy(word_dim, hidden_dim)
np.random.seed(10)


learning_rate=0.001
nepoch=10
evaluate_loss_after=1
num_examples_seen = 0
losses = []
for epoch in range(nepoch):
    if epoch % evaluate_loss_after == 0:
        loss = rnn.calculate_losss(X_train, Y_train)
        losses.append((num_examples_seen, loss))
        print("epoch=%d: training loss=%f" % (epoch, loss))
        if len(losses) > 1 and losses[-1][1] > losses[-2][1]:
            learning_rate *= 0.5
            print("Setting learning rate to %f" % learning_rate)
        sys.stdout.flush()
    for i in range(len(Y_train)):
        rnn.SGD_gradient(X_train[i], Y_train[i], learning_rate)
        num_examples_seen += 1


# save = save_model_("model.npy", rnn)
