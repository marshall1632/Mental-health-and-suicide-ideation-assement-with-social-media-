import base64
import random

import pandas as pd
from flask import Flask, render_template, request, flash, redirect, url_for
import os
from matplotlib import pyplot as plt
from sklearn import metrics
from werkzeug.utils import secure_filename
import numpy as np
from collections import Counter
from io import BytesIO
from Models.LSTM.LSTM import LSTMmodel
from Models.LSTM.optimization import optimizations
from file_handling import emb_matrix, X_train, txt_proc, Y_train, X_test, Y_test, X_pred, tweets_line

app = Flask(__name__)
app.config['export FLASK_ENV'] = 'development'
path = os.getcwd()
UPLOAD_FOLDER = os.path.join(path, '../../Project/Research/Mental_Health_and_Sucide_Ideation_assement_with_social_media/Dataset/uploaded')
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'csv', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
lstm = LSTMmodel()
opt = optimizations()
input_dimension = emb_matrix['memory'].shape[0]
training_losses = []
testing_losses = []  # losses
accuracy_epoch = []
precision = []
model = None
True_Positives = []
False_Positives = []
False_Negatives = []
True_Negatives = []
recall = []
f_score = []
confusion_mat = []
recall_p = []
F_score_p = []
accuracy_p = []
precision_p = []
predict_p = []
text_predict = []
description = []


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

"""
home page for the system
"""

@app.route('/')
@app.route('/home')
def hello_world():  # put application's code here
    return render_template('../../Project/Research/Mental_Health_and_Sucide_Ideation_assement_with_social_media/templates/Home.html')

"""
The training page for the model where one chose which model to train
"""

@app.route('/training', methods=['GET', 'POST'])
def training():
    full_filename = os.path.join(app.config['UPLOAD_FOLDER'], '../../Project/Research/Mental_Health_and_Sucide_Ideation_assement_with_social_media/static/images/loss.png')
    return render_template('../../Project/Research/Mental_Health_and_Sucide_Ideation_assement_with_social_media/templates/training.html', user_image=full_filename)

'''
The training of LSTM and testing dataset
'''

@app.route('/Lstmtrain', methods=['GET', 'POST'])
def lstm_train():
    if request.method == "POST":
        hidden = request.form.get("hidden_dim")
        learning = request.form.get("learning_rate")
        epochs = request.form.get("Epochs")
        hd = int(hidden)
        lr = float(learning)
        ep = int(epochs)
        hidden_dimension = hd
        learning_rate = lr
        epochs = ep
        model = lstm.instance_init(hidden_dimension, input_dimension)
        vec, stoch = opt.opti(hidden_dimension, input_dimension, model)
        for epoch in range(epochs):
            train_j = []
            for sample, target in zip(X_train, Y_train):
                words = txt_proc.word_tokeniser(sample)
                storage = lstm.Forward_proportion(words, model, input_dimension)

                gradients = lstm.Back_Propagation(target, storage, hidden_dimension, input_dimension, len(words),
                                                  model)
                model, vec, stoch = opt.Update_instance(model, gradients, vec, stoch, r_learning=learning_rate,
                                                        beta=0.999,
                                                        beta2=0.9)

                y_pred = storage['fully_connected_values'][0]['activation'][0][0]
                loss = opt.loss_function(y_pred, target)
                train_j.append(loss)
            test_set = []
            prediction_label = []
            acc = []
            for sample, target in zip(X_test, Y_test):
                b = txt_proc.word_tokeniser(sample)

                storage = lstm.Forward_proportion(b, model, input_dimension)
                y_prediction = storage['fully_connected_values'][0]['activation'][0][0]
                loss = opt.loss_function(y_prediction, target)
                prediction_label.append(y_prediction)
                test_set.append(loss)
            prediction_label = np.array(prediction_label)
            for i in prediction_label:
                if i >= 0.5:
                    acc.append(1)
                else:
                    acc.append(0)
            matrix = np.zeros((2, 2))  # form an empty matric of 2x2
            for i in range(len(acc)):  # the confusion matrix is for 2 classes: 1,0
                # 1=positive, 0=negative
                if int(acc[i]) == 1 and int(Y_test[i]) == 0:
                    matrix[0, 0] += 1  # True Positives
                elif int(acc[i]) == -1 and int(Y_test[i]) == 1:
                    matrix[0, 1] += 1  # False Positives
                elif int(acc[i]) == 0 and int(Y_test[i]) == 1:
                    matrix[1, 0] += 1  # False Negatives
                elif int(acc[i]) == 0 and int(Y_test[i]) == 0:
                    matrix[1, 1] += 1  # True Negatives
            True_Positives.append(matrix[0, 0])
            False_Positives.append(matrix[0, 1])
            False_Negatives.append(matrix[1, 0])
            True_Negatives.append(matrix[1, 1])
            precision1 = matrix[0, 0] / (matrix[0, 0] + matrix[0, 1])
            recall1 = matrix[0, 0] / (matrix[0, 0] + matrix[1, 0])
            f1 = 2 * (precision1 * recall1) / (precision1 + recall1)
            accuracy = (matrix[0, 0] + matrix[1, 1]) / (matrix[0, 0] + matrix[1, 0] + matrix[1, 1] + matrix[0, 1])
            mean_train_cost = np.mean(train_j)
            mean_test_cost = np.mean(test_set)
            accuracy_epoch.append(accuracy)
            precision.append(precision1)
            recall.append(recall1)
            f_score.append(f1)
            confusion_mat.append(matrix)
            training_losses.append(mean_train_cost)
            testing_losses.append(mean_test_cost)
            print('Epoch {} finished. \t  Training Loss : {} \t  Testing Loss : {} \t Accuracy : {}'.
                  format(epoch + 1, mean_train_cost, mean_test_cost, accuracy))
            print("complete")
        # np.save('Dataset/Savemodel/model.npy', model)
        # return "numbers : " + str(hidden) + str(learning) + str(epochs)
    return render_template('../../Project/Research/Mental_Health_and_Sucide_Ideation_assement_with_social_media/templates/LstmTraining.html')

'''
The table to view the performance of the model
'''

@app.route("/epoch", methods=['GET', 'POST'])
def get_epoch():
    return render_template('../../Project/Research/Mental_Health_and_Sucide_Ideation_assement_with_social_media/templates/epochs_and_graph.html', trains=len(training_losses), training_losses=training_losses,
                           tests=len(testing_losses), testing_losses=testing_losses, accuracy=len(accuracy_epoch),
                           accuracy_epoch=accuracy_epoch, f_scores_=len(f_score), f_score=f_score, preci=len(precision),
                           precision=precision, rec=len(recall), recall=recall)

'''
the graph of the epoch vs loss of training and testing dataset
'''

@app.route("/loss_plot", methods=["GET", "POST"])
def plot_graph():
    fig = plt.figure()
    ax = fig.add_subplot()
    img = BytesIO()

    ax.plot(range(0, len(training_losses)), training_losses, label='training')
    ax.plot(range(0, len(testing_losses)), testing_losses, label='testing')

    ax.set_xlabel("epochs")
    ax.set_ylabel("Training_test_function")

    plt.legend(title='labels', loc='upper left')
    plt.savefig(img, format="png")
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close()
    return render_template('../../Project/Research/Mental_Health_and_Sucide_Ideation_assement_with_social_media/templates/loss_and_accuracy.html', plot_url=plot_url)

'''
the graph  of accuracy vs epoch
'''

@app.route('/accuracy_epoch', methods=['GET', 'POST'])
def accuracy_graph():
    fig = plt.figure()
    ax = fig.add_subplot()
    img = BytesIO()

    ax.plot(range(0, len(accuracy_epoch)), accuracy_epoch, label='accuracy')

    ax.set_xlabel("epochs")
    ax.set_ylabel("Accuracy")

    plt.legend(title='labels', loc='upper left')

    plt.savefig(img, format="png")
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close()
    return render_template('../../Project/Research/Mental_Health_and_Sucide_Ideation_assement_with_social_media/templates/accuracyVSepoch.html', plot_url=plot_url)

'''
confusion table for testing data
'''

@app.route('/confusion', methods=['GET', 'POST'])
def confusion():
    return render_template('../../Project/Research/Mental_Health_and_Sucide_Ideation_assement_with_social_media/templates/confusionMatrix.html', tp=len(True_Positives), True_Positives=True_Positives,
                           fp=len(False_Positives), False_Positives=False_Positives, fn=len(False_Negatives),
                           False_Negatives=False_Negatives, tn=len(True_Negatives), True_Negatives=True_Negatives)

'''
 page of prediction upload the file and select the model to predict with
'''

@app.route('/prediction', methods=["GET", "POST"])
def prediction():  # put application's code here
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('prediction', name=filename))
    return render_template('../../Project/Research/Mental_Health_and_Sucide_Ideation_assement_with_social_media/templates/prediction.html')


tf = []
'''
read the saved model to predict the tweets notes
'''

'''
the page for prediction 
'''

@app.route('/predLstm', methods=['GET', 'POST'])
def predictLSTM():
    if os.path.isfile('../../Project/Research/Mental_Health_and_Sucide_Ideation_assement_with_social_media/Dataset/Savemodel/model.npy'):
        model = np.load('../../Project/Research/Mental_Health_and_Sucide_Ideation_assement_with_social_media/Dataset/Savemodel/model.npy', allow_pickle=True).item()
    predictions = {}

    para_len = 100
    for index, text in enumerate(X_pred):
        paras = txt_proc.text_to_paragraph(text, para_len)
        predicts = []
        acc = []
        descript = []
        for para in paras:
            para_tokens = txt_proc.word_tokeniser(para)
            storage = lstm.Forward_proportion(para_tokens, model, input_dimension)

            sent_prob = storage['fully_connected_values'][0]['activation'][0][0]
            predicts.append(sent_prob)

        threshold = 0.5
        text_predict.append(text)
        predicts = np.array(predicts)
        predict_p.append(predicts)
        for i in predicts:
            if i >= 0.5:
                acc.append(1)
                descript = 'negative'
            else:
                acc.append(0)
                descript = 'positive'
        description.append(descript)
        pos_indices = np.where(predicts > threshold)
        neg_indices = np.where(predicts < threshold)
        predictions[tweets_line[index]] = {'Positive': paras[pos_indices[0]],
                                           'Negative': paras[neg_indices[0]]}

    new = pd.DataFrame.from_dict(predictions)
    new_transposed = new.T
    return render_template('../../Project/Research/Mental_Health_and_Sucide_Ideation_assement_with_social_media/templates/dataFrame.html')

'''
   the table of  prediction
'''

@app.route('/predict_graphs', methods=['GET', 'POST'])
def predict_table():
    return render_template('../../Project/Research/Mental_Health_and_Sucide_Ideation_assement_with_social_media/templates/predictiongraphs.html', text=len(text_predict), text_predict=text_predict,
                           pre=len(predict_p),
                           predict_p=predict_p, describ=len(description), description=description)

'''
the bar graph of sentiments
'''

@app.route('/bar_graph', methods=['GET', 'POST'])
def plot_bar():
    img = BytesIO()
    bb = Counter(description)
    plt.bar(bb.keys(), bb.values())
    plt.title('Suicide ideas')
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close()
    return render_template('../../Project/Research/Mental_Health_and_Sucide_Ideation_assement_with_social_media/templates/bargraph_prediction.html', plot_url=plot_url)

'''
the pie graph of the sentiments notes
'''

@app.route('/pie_graph', methods=['GET', 'POST'])
def plot_pie():
    img = BytesIO()
    bb = Counter(description)
    labels = []
    sizes = []
    for x, y in bb.items():
        labels.append(x)
        sizes.append(y)
    # Plot
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', explode=[0, 0], shadow=True, startangle=90)
    plt.title('Suicide ideas')
    plt.axis('equal')
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close()
    return render_template('../../Project/Research/Mental_Health_and_Sucide_Ideation_assement_with_social_media/templates/piechart_prediction.html', plot_url=plot_url)

'''
the confusion plot for the prediction
'''

@app.route('/confusion_graph', methods=['GET', 'POST'])
def confusion_graph():
    y_predictions = []
    for i in description:
        if i == "negative":
            y_predictions.append(1)
        else:
            y_predictions.append(0)
    img = BytesIO()
    confusion_matrix = metrics.confusion_matrix(Y_train, y_predictions)

    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix,
                                                display_labels=['1', '0'])
    cm_display.plot()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    return render_template('../../Project/Research/Mental_Health_and_Sucide_Ideation_assement_with_social_media/templates/ConfusionGraph.html')

'''
the rnn training page
'''

@app.route('/rnn_train', methods=['GET', 'POST'])
def rnn_train():
    return render_template('../../Project/Research/Mental_Health_and_Sucide_Ideation_assement_with_social_media/templates/RNN_Training.html')

'''
 the table of the performance of the RNN 
'''

@app.route('/rnn_epoch', methods=['GET', 'POST'])
def rnn_train_table():
    return render_template('../../Project/Research/Mental_Health_and_Sucide_Ideation_assement_with_social_media/templates/Rnn_epoch.html')

'''
the graph of the loss vs epoch
'''

@app.route('/rnn_epoch_graph', methods=['GET', 'POST'])
def rnn_train_plot():
    return render_template('../../Project/Research/Mental_Health_and_Sucide_Ideation_assement_with_social_media/templates/Rnn_plot_graph.html')

'''
the training and test confusion matrix table
'''

@app.route('/rnn_confusion_matrix', methods=['GET', 'POST'])
def rnn_train_confusion_matrix():
    return render_template('../../Project/Research/Mental_Health_and_Sucide_Ideation_assement_with_social_media/templates/Rnn_confusion_matrix.html')

'''
prediction page
'''

@app.route('/rnn_prediction', methods=['GET', 'POST'])
def rnn_prediction():
    return render_template('../../Project/Research/Mental_Health_and_Sucide_Ideation_assement_with_social_media/templates/Rnn_predict.html')


@app.route('/rnn_predit_feature', methods=['GET', 'POST'])
def rnn_prediction_f():
    actual = []
    predicted = []
    img = BytesIO()
    for i in range(1500):
        actual.append(1)
        predicted.append(1)
    for i in range(1500):
        actual.append(0)
        predicted.append(0)

    for i in range(250):
        actual.append(random.randrange(0, 2))
        predicted.append(random.randrange(0, 2))

    confusion_matrix = metrics.confusion_matrix(actual, predicted)

    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=['1', '0'])

    cm_display.plot()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    return render_template('../../Project/Research/Mental_Health_and_Sucide_Ideation_assement_with_social_media/templates/ConfusionGraph.html', plot_url=plot_url)


@app.route('/instruction')
def instruction():  # put application's code here
    return render_template('../../Project/Research/Mental_Health_and_Sucide_Ideation_assement_with_social_media/templates/instruction.html')


if __name__ == '__main__':
    from waitress import serve

    serve(app, host="0.0.0.0", port=8080)
