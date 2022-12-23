# import collections
# import os
#
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import pylab
# import seaborn as sns
# from sklearn import metrics
# from sklearn.metrics import confusion_matrix
# from Models.LSTM.LSTM import LSTMmodel
# from Models.LSTM.optimization import optimizations
# from file_handling import emb_matrix, X_train, txt_proc, Y_train, X_test, Y_test, X_pred, tweets_line
#
# lstm = LSTMmodel()
# opt = optimizations()
# hidden_dimension = 64
# input_dimension = emb_matrix['memory'].shape[0]
# learning_rate = 0.001
# epochs = 5
# model = lstm.instance_init(hidden_dimension, input_dimension)
# vec, stoch = opt.opti(hidden_dimension, input_dimension, model)

# training_losses = []
# testing_losses = []
# accuracy_epoch = []
# precision = []
# recall = []
# f_score = []
# confusion_mat = []
# prediction_label = []
#
#
# def Training_test_function(instance, vec, stoch):
#     for epoch in range(epochs):
#         train_j = []
#         for sample, target in zip(X_train, Y_train):
#             words = txt_proc.word_tokeniser(sample)
#             storage = lstm.Forward_proportion(words, instance, input_dimension)
#
#             gradients = lstm.Back_Propagation(target, storage, hidden_dimension, input_dimension, len(words),
#                                               instance)
#             instance, vec, stoch = opt.Update_instance(instance, gradients, vec, stoch, r_learning=learning_rate,
#                                                        beta=0.999,
#                                                        beta2=0.9)
#
#             y_pred = storage['fully_connected_values'][0]['activation'][0][0]
#             loss = opt.loss_function(y_pred, target)
#             train_j.append(loss)
#         test_set = []
#         pre_y = []
#         acc = []
#         for sample, target in zip(X_test, Y_test):
#             b = txt_proc.word_tokeniser(sample)
#
#             storage = lstm.Forward_proportion(b, instance, input_dimension)
#             y_prediction = storage['fully_connected_values'][0]['activation'][0][0]
#             loss = opt.loss_function(y_prediction, target)
#             pre_y.append(y_prediction)
#             test_set.append(loss)
#         prediction_label = pre_y
#         prediction_label = np.array(prediction_label)
#         for i in prediction_label:
#             if i >= 0.5:
#                 acc.append(1)
#             else:
#                 acc.append(0)
#         matrix = np.zeros((2, 2)).astype(int)
#         for i in range(len(acc)):  # the confusion matrix is for 2 classes: 1,0
#             # 1=positive, 0=negative
#             if int(acc[i]) == 1 and int(Y_test[i]) == 0:
#                 matrix[0, 0] += 1  # True Positives
#             elif int(acc[i]) == -1 and int(Y_test[i]) == 1:
#                 matrix[0, 1] += 1  # False Positives
#             elif int(acc[i]) == 0 and int(Y_test[i]) == 1:
#                 matrix[1, 0] += 1  # False Negatives
#             elif int(acc[i]) == 0 and int(Y_test[i]) == 0:
#                 matrix[1, 1] += 1  # True Negatives
#         precision1 = matrix[0, 0] / (matrix[0, 0] + matrix[0, 1])
#         recall1 = matrix[0, 0] / (matrix[0, 0] + matrix[1, 0])
#         f1 = 2 * (precision1 * recall1) / (precision1 + recall1)
#         accuracy = (matrix[0, 0] + matrix[1, 1]) / (matrix[0, 0] + matrix[1, 0] + matrix[1, 1] + matrix[0, 1])
#         mean_train_cost = np.mean(train_j)
#         mean_test_cost = np.mean(test_set)
#         accuracy_epoch.append(accuracy)
#         precision.append(precision1)
#         recall.append(recall1)
#         f_score.append(f1)
#         confusion_mat.append(matrix)
#         training_losses.append(mean_train_cost)
#         testing_losses.append(mean_test_cost)
#         print('Epoch {} finished. \t  Training Loss : {} \t  Testing Loss : {} \t Accuracy : {}'.
#               format(epoch + 1, mean_train_cost, mean_test_cost, accuracy))
#
#
# print(Training_test_function(model, vec, stoch))
# print("Recall: ", recall)
# print("precision: ", precision)
# print("F_score: ", f_score)
# print(confusion_mat)
#
# # # np.save('Dataset/Savemodel/model.npy', model)
# #
# #
#
# def plot_graph():
#     fig = plt.figure()
#     ax = fig.add_subplot()
#
#     ax.plot(range(0, len(training_losses)), training_losses, label='training')
#     ax.plot(range(0, len(testing_losses)), testing_losses, label='testing')
#     ax.plot(range(0, len(accuracy_epoch)), accuracy_epoch, label='accuracy')
#
#     ax.set_xlabel("epochs")
#     ax.set_ylabel("Training_test_function")
#
#     plt.legend(title='labels', loc='upper left')
#     # plt.savefig('static/images/loss.png')
#     return plt.show()
#
#
# plot_graph()


# def plot():
#     matrix = confusion_mat
#
#     pylab.figure()
#     pylab.imshow(matrix, interpolation='nearest', cmap=pylab.cm.jet)
#     pylab.title("Confusion Matrix")
#
#     for i, vi in enumerate(matrix):
#         for j, vj in enumerate(vi):
#             pylab.text(j, i+.1, "%.1f" % vj, fontsize=12)
#             pylab.colorbar()
#
#             # classes = np.arange(len(labels))
#             # pylab.xticks(classes, labels)
#             # pylab.yticks(classes, labels)
#
#             pylab.ylabel('Expected label')
#             pylab.xlabel('Predicted label')
#
#
#             return pylab.show()
# plot()
# def plot_matrix(y_test, y_pred):
#     cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
#     ax = plt.subplot()
#     sns.heatmap(cm, annot=True, ax=ax, fmt='g')  # annot=True to annotate cells
#     # labels, title and ticks
#     ax.set_xlabel('Predicted', fontsize=20)
#     ax.xaxis.set_label_position('top')
#     ax.xaxis.set_ticklabels([0, 1], fontsize=15)
#     ax.xaxis.tick_top()
#
#     ax.set_ylabel('True', fontsize=20)
#     ax.yaxis.set_ticklabels(['spam', 'ham'], fontsize=15)
#     plt.show()
#
#
# plot_matrix(Y_test, prediction_label)


#plot_graph()
# recall_p = []
# F_score_p = []
# accuracy_p = []
# precision_p = []
# predict_p = []
# text_predict = []
# description = []
# predictions = {}
# boolean = []
# para_len = 10
# y_predictions = []
#
# if os.path.isfile('Dataset/Savemodel/model.npy'):
#     model = np.load('Dataset/Savemodel/model.npy', allow_pickle=True).item()
# for index, text in enumerate(X_pred):
#     paras = txt_proc.text_to_paragraph(text, para_len)
#     predicts = []
#     acc = []
#     descript = []
#     for para in paras:
#         para_tokens = txt_proc.word_tokeniser(para)
#         storage = lstm.Forward_proportion(para_tokens, model, input_dimension)
#
#         sent_prob = storage['fully_connected_values'][0]['activation'][0][0]
#         predicts.append(sent_prob)
#
#     threshold = 0.5
#     text_predict.append(text)
#     predictss = np.array(predicts)
#     predict_p.append(predictss)
#     for i in predicts:
#         if i >= 0.5:
#             acc.append(1)
#             descript = 'positive'
#         else:
#             acc.append(0)
#             descript = 'negative'
#     description.append(descript)
#     boolean.append(acc)
#     pos_indices = np.where(predictss < threshold)
#     print(pos_indices)
#     neg_indices = np.where(predictss > threshold)
#     print(neg_indices)
#     predictions[tweets_line[index]] = {'Positive': paras[pos_indices[0]],
#                                        'Negative': paras[neg_indices[0]]}
#
# print(predictions)

# for i in description:
#     if i == "negative":
#         y_predictions.append(1)
#     else:
#         y_predictions.append(0)
#
#
# def matrix_plot(Y_test, X_pred):
#     confusion_matrix = metrics.confusion_matrix(Y_test, X_pred)
#
#     cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix,
#                                                 display_labels=['0', '1'])
#     cm_display.plot()
#     plt.savefig('static/confusion.png')
#     return plt.show()
#
#
# matrix_plot(Y_train, y_predictions)

# x_axis = []
# # m_ = []
# data = {'positive sentiment': [], 'negative sentiment': []}
# for tweet_t in predictions:
#     m_.append(tweet_t)
#     num_neg = len(description[tweet_t]['Positive'])

# for tweet_l in predictions:
#     x_axis.append(tweet_l)
#     no_pos_paras = len(predictions[tweet_l]['Positive'])
#     no_neg_paras = len(predictions[tweet_l]['Negative'])
#     pos_perc = no_pos_paras / (1 + no_neg_paras)
#     data['positive sentiment'].append(pos_perc * 100)
#     data['negative sentiment'].append(100 * (1 - pos_perc))

# index = pd.Index(x_axis, name='target')
# df = pd.DataFrame(data, index=index)
# ax = df.plot(kind='bar', stacked=True)
# ax.set_ylabel('percentage')
# ax.legend(title='labels', loc='upper right')
# plt.savefig('static/images/prediction.png')
# plt.show()
# print("complete........")

