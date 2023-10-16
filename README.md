# Mental-health-and-suicide-ideation-assement-with-social-media-
final year project to check for personal tweets timeline of individual and tell is that particular individual is depressed or not. On the dataset one need to add the glove 300d file in order to train the model.


Mental Health and Suicide Ideation Assessment with social media
M.M. Nekhubvi1[215000028]
1 Academy of Computer Science and Software Engineering
University of Johannesburg South Africa
215000028@student.uj.ac.za
Abstract Text classification is the main function in the turf of natural language processing. The are many procedures used in text classification, e.g., machine learning (ML), deep learning (DL) etc., but all these methods still need to be improved due to the strong relevance of context and complexity abstraction of text semantic information. In this paper, we will examine deep learning methods in text classification. The model to have been examine is Long Short-Term Memory (LSTM)-based classification development that differentiates among suicidal and non-suicidal language. The technique for detecting the suicidal speech on Twitter combines word embeddings with LSTM neural network. The paper is going to look at the implementation of LSTM and word embedding GloVe working together to classify the sentiments if positive or negative.
Keywords: Text Classification, Natural language processing, ML, DL, LSTM, RNN, CNN, Word embedding, GloVe, TI-DF.
1. Introduction
Artificial Intelligence (AI) is related to high-tech automation, science fiction, and robotics and can appear overwhelming to a few groups of people. AI, in general, has developed significantly over the past years and is the current driving force behind 4IR. In contrast, it proposes an enhancement to most fields that need machines, in-volving ethics, social, medical science, and legitimate challenges associated with its growth. The paper will examine how AI impact the mental health and suicidal ideation assessment with social media. An article was made by World Health Organization in recent times that suicide arose every 40 seconds worldwide, with 804,000 suicide deaths occurring in 2012 [1] The issue is not only in South Africa but the whole world. Whilst both ML and DL techniques are dominantly used for many fields of studies, they also impact to solving some of the biggest social difficulties human encounter worldwide [2].
Recently, Deep learning have attained magnificent results in natural language pro-cessing and have marvellous capabilities of learning [3]. The natural language pro-cessing makes use of Recurrent Neural Network (RNN), Long Short-Term Memory (CNN), Convolutional Neural Network (CNN), etc. to detect the suicidal ideation or suicidal notes [4]. CNN neural network: it is created along with the strong ability of picture recognition [5] However, at the current time, CNN covers a broad range of multiple text identification responsibilities and gives incredible outcomes. CNN is applied to organized and well-structured text; the simulation will learn and discover relationships that are most likely to be lost from the feed-forward network. For example, a word like "down" in a sentence of "feel down" and "down to earth" has a different meaning. Feed-forward Neural Networks are like CNN, where the nodes do not form due to connection. LSTM is part of RNN architectures used in deep learning to help predict a series of sentences and classify the process. LSTM has memory cells that help control the flow, whereas RNN is more robust to capture long-term depend-ency. And prevent explosion gradients that are often seen in RNN models. Word Embedding is a set of dialect modeling and highlights learning procedures in NLP. It is a hybrid input layer of CNN and LSTM that changes words into a representation of a real-valued vector. The technique lets the lexicon's words tend to map into a specific vector space in a low dimensional space [6]. The model uses the unsupervised training fundamental to issue tasks for solving supervised learning.
The paper will consist of several sections. First section we will look at the problem statement, next we look at the literature review which focuses on the related work of suicidal ideation using the deep learning models. The third section will examine the experiment of how the system is going to operate. Section four focuses on the results of the experiment. Then lastly, we conclude the paper. Next, we look at the problem statement and the purpose of the paper.
2. Problem Background and Purpose
Suicide ideation refers to planning or thinking about suicide. The idea can range from designing a detailed plan to having a short consideration but does not include the final act of suicide. In the recent time much research conducted into text classification of suicide notes or detection of suicidal ideation online [7]. Researchers used various methodologies of traditional ML, DL, and sentimental analysis [8]. The research is conducted in different area such as linguistic, healthcare or psychology. Many applications are developed to compare different types of textual data containing suicidal ideation such as depressed language, loneliness thoughts, etc. [9]. Altogether the is growing interest in studying the content created online that seek need for help or detecting mental health issues [7].
In the paper we will provide the overview of DL technique in mental health and suicide ideation, the examination of recent literature on DL and ML in mental healthcare (results and methods).
3. Literature Review
Before the text classification of the models can be discussed four area need to be briefly addressed to lay out the basis of the rest of the paper. This area includes, what is text classification and Natural Language processing, the
use of text classifications while utilizing the traditional ML, what is Word Embedding and other feature extraction, and finally use of different DL techniques such as the neural network
A. Text classification and NLP
As mentioned, that text classification is the primary foundation in the field of natural language processing. Also known as text categorization or text tagging is the pro-cess of categorizing text into groups with the use of NLP, text classifiers can automatically analyse text and then assign a set of pre-defined categories or tags based on its content [2].
Natural Language Processing (NLP) is a subdomain of AI that allows the computer to process algorithms and analyse human language within the shape of unstructured content and includes semantic understanding, information extraction, and language translation [10]. NLP is heavily used in mental health practice before performing other AI techniques because of the data from clinical notes or written language and conversations. The capacity of a computer algorithm to naturally get the underlying words, despite the human dialect, is a massive progression in innovation and fundamental for mental healthcare implementation [11]. To be more effective in recognizing suicidal tendencies, we need regular language patterns in a social media text. Many NLP techniques are used to support detecting suicidal notes.
B. Traditional Machine Learning Models
ML models' technique is used to tackle various real-world issues such as computer vision, voice localization, machine translation, handwriting recognition, etc. [12]. We are looking at the ML model to compare how ML works in various systems. One of the most used ML models is the Support Vector Machine (SVM). It is a supervised learning model connected to the learning algorithms that analyse data used for regression analysis and identification. SVM performs a nonlinear identification using the kernel technique. Its implementation avoids and minimizes the identification of errors. A kernel technique is an action done to the coach set of data to produce good results. The data set is allowed by the increments of dimension done by kernel function [13].
The limitation of ML in text classification is lack of data. When the model is given a poor dataset then it will only give poor results. So, all this is being solved by the neural network since they are data-eating machines that need plentiful numbers of training dataset. Most of ML models make use of the traditional statistical methods and can stop dead if they do not see the model interpretable.
C. Feature Extraction
They are various text feature extraction approaches used in classifying sentences into categories with neural network to perform high accuracy [14]. The feature extraction methods like Term Frequency Inverse Document Frequency (TF-IDT), Word Embedding which outperform other methods with magnificent accuracy. TF-IDF is a statistical measure that studies how relevant a word is to a document in a collection of documents. It works by increasing proportionally the number of times a word appears in a document. TF-IDF for word in a dataset is calculated by multiplying two metrics the term frequency and the inverse document frequency of words on the dataset [14].
The word embedding Is a set of dialect modeling and highlights learning procedures in NLP. It is a hybrid input layer of CNN and LSTM that changes words into a representation of a real-valued vector. The technique allows the lexicon's words to map into a specific vector space in a low dimensional space [15]. The model uses the unsupervised training fundamental to issue tasks for solving supervised learning. The text is a set of sequences 𝑥1, 𝑥2, 𝑥3, ..., 𝑥𝑡, which is characterized by an index number that is converted to a low dimension to transform indices into d-dimension of embed-ding vector 𝑋𝑡∈𝑅𝑑 When training [15]. They are different technique to create word embedding through the neural networks, but the study will look closely at the Glove approach that was introduces by penning et al in 2014 [16] and depends on establishment of a global co-occurrence matrix of words in the corpus. The embedding vector are built on the analysis of two more words occurring simultaneously of word in a window [17].
D. Deep Learning Techniques
In recent times neural network models in NLP have shown a significant improvement in detecting suicide ideation from the implementations that are complex deep learning computing to reduce the traditional ML systems. Recurrent neural network (RNN) is implemented to outline a group of modeling. The effectiveness of the LSTM model is the most used for long-range dependency. Sawhney et al. showed the ability and strength of the C-LSTM model, and similarities were checked between the ML and DL identifier and showed better results [18]. Ji et al. also used LSTM and compared them to the other five ML models, and his study shows a significant benchmark for detecting suicidal ideation on Twitter and Reddit [19].
LSTM is part of RNN architectures used in deep learning to help predict a series of sentences and classify the process. LSTM has memory cells that help control the flow, whereas RNN is more robust to capture long-term dependency. And prevent explosion gradients that are often seen in RNN models.
This layer is a subset of the CNN neural network; it is created with the strong ability of picture recognition [20]. However, at the current time, CNN covers a wide range of multiple text identification tasks and gives incredible results. CNN is applied to organized and well-structured text; the model will learn and discover patterns that are most likely to be lost from the feed-forward network. For example, a word like "down" in a sentence of "feel down" and "down to earth" has a different meaning. Feed-forward Neural Networks are like CNN, where the nodes do not form due to connection. Zhang et al. work in convolution layer examine in a simple form where a single neuron in CNN has a region containing an input sample of text and image.
3. Experiment
In this section we will discuss all the methods/ techniques used to conduct the mental health system. First, we will discuss the datasets how and where the data is obtained. Next the study will look at the technique used to conduct the classification of the system. Then we will look at how the system works along with the results of that are obtain by each neural network that was built for the system.
A. Datasets
First, we describe the features of Twitter because they are more important to under-stand the context of our research. Many of the features of an internet forum may be found on Twitter where members can post links or text messages. The tweets are ar-ranged by subject of topics or hashtags, which are smaller communities. Along with tweets user can participate by commenting on the post and retweet the post in response to the other user posts.
Data collection: The datasets used in the system for training and testing were found in different platform then combine into one file to use. First collection of data was random tweets can be sourced from the sentiment140 dataset available on Kaggle. Then another comes from the tweets collected on Linux system commands using Twint tool is available on: https://drive.google.com/drive/folders/1z-PrTTT6u3xciSUc0eZQRfQa4qn09urc?usp=sharing where the is separated depression, suicide, loneliness, etc. The captured information was analysed to reveal social inter-actions.
B. Pre-processing
Before creating any deep learning model, pre-processing data is necessary step. The steps that were undertaken to clean the data and convert it to its numerical representation. The data then cleaned by removing all unhelpful noise from the data by removing html, brackets, stop words and converting all character to lowercase this is done so to avoid the cluster of words that computer cannot understand. Then last step was to convert the words to vector, were the system take words that have the same mean-ing are represented similarly in a word embedding, which is learnt representation for text. In a predetermined vector space, each word is represented as real-values vectors. Glove is a Stanford-developed unsupervised technique for creating word embeddings by a corpus’s global word-word co-occurrence matrix. The embedding dataset is available on: https://nlp.stanford.edu/projects/glove the system uses the glove.6B.300d.txt which uses the least amount of memory. A collection of GloVe word embeddings was trained using billion of tokens, up to 840 billion tokens in certain cases [21].
C. Building the Deep Learning Model
Variable-length sequence processing has been extensively employed using Recurrent Neural Network (RNN) [15]. But because a conventional RNN is equal to a multi-layer feed-forward neural network, a significant amount of historical data delivered by long sequences would result in diminishing gradient and data loss. The neural network that has been enhanced and built on RNN is called Long Short-Term Memory (LSTM) as discussed above LSTM successfully preserves the history data in length sequences, preventing gradient disappear and information loss from too much layer RNN training using memory cell and the three control gates.
Fig. 1. LSTM structure [22]
In figure 1, the structure of LSTM is shown. The architecture includes the three gates-the input gates, forget gate, output gate- respectively govern updating erasing, and output of data history, and the memory cell for storing data history. The input gate controls how the incoming vectors affect the memory cell state. The memory cell can influence the output gate. The memory cell’s ability is its prior knowledge is controlled by the forget gate. The number of LSTM hidden layer is H when the input sequences are 𝑋=[𝑥1; 𝑥2;…;𝑥𝑇], where 𝑥𝑡 is the d-dimension word embedding vector. The gates are changed as follows at time step t:
𝑖𝑡= 𝜎(𝑊𝑖𝑥𝑡 + 𝑈𝑖ℎ𝑡−1+ 𝑏𝑖) (1)
𝑜𝑡= 𝜎(𝑊𝑜𝑥𝑡 + 𝑈𝑜ℎ𝑡−1+ 𝑏𝑜) (2)
𝑓𝑡= 𝜎(𝑊𝑓𝑥𝑡 + 𝑈𝑓ℎ𝑡−1+ 𝑏𝑓) (3)
𝐶𝑡= 𝑓𝑡⨂𝐶𝑡−1+tanh(𝑊𝑐𝑥𝑡 + 𝑈𝑐ℎ𝑡−1+ 𝑏𝑐) (4)
ℎ𝑡= 𝑜𝑡⨂tanh (𝐶𝑡) (5)
The input vector, input gate, and forget gate together define the state of the memory cell at time step t, which is 𝐶𝑡. The LSTM’s final output has a dimension equal to the number of hidden layer nodes, which is ℎ𝑡. the sigmoid function is H. These are the network parameters: W ∈ 𝑅𝐻×𝐸 , b ∈ 𝑅𝐻×1, U ∈ 𝑅𝐻×𝐻 [21].
A. Construction of Environment and parameter setting
Python is used as a programming language and the system make use of FLASK micro web framework written in python. The experiment environment is shown as follow.
TABLE I. Environment lab
Software and Hardware
Configure
CPU
AMD Ryzen 5 5600H with Radeon Graphics 3.30 GHz
RAM
8.00 GB
GPU
NVIDIA GeForce GTX 1650
Operating System
Windows 11
Development Environment
PyCharm 2022.1.3
The optimization strategy used to update the parameter is Adam optimization, and the learning rate is 0.001. The dimension of the word embedding is 300. The number of hidden dimensions is 64. Finally,
the final classification model is obtained with 10 epochs. Although the system allows the user to train themselves and they can put their own parameter to train the dataset.
4. Results and Discussion
The LSTM model is evaluated to produce quality text classification task, the use of classification accuracy, F-score, recall, and precision are used to measure the performance of the model quality. One of the regularly used metrics in the turf of text categorization are the classification mentioned above. As illustrated in Table 4, we often create a matrix based on the categorization outcome below.
Table II. Confusion matrix
Suicidal note
Non- Suicidal note
Positive
True Positives (TP)
False Positive (FP)
Negative
False Negative (FN)
True Negative (TN)
The percentage of the text that falls into the right category as measured by classifications above and expressed by the following formula:
𝑎𝑐𝑐𝑢𝑟𝑎𝑐𝑦= 𝑇𝑃 + 𝑇𝑁𝑇𝑃 + 𝑇𝑁 + 𝐹𝑁 + 𝐹𝑃 (6)
𝑅𝑒𝑐𝑎𝑙𝑙 = 𝑇𝑃𝑇𝑃+ 𝐹𝑁 (7)
𝑃𝑟𝑒𝑐𝑖𝑠𝑖𝑜𝑛= 𝑇𝑃𝑇𝑃 + 𝐹𝑃 (8)
𝐹−𝑠𝑐𝑜𝑟𝑒= 2𝑇𝑃2𝑇𝑃 + 𝐹𝑃 + 𝐹𝑁 (9)
All the classifications mentioned are used to validate the model performance on test set vs the training set. The outcome illustrated in Table 5 show the performance of LSTM model with different parameters.
Table III. Experiment Results
Epoch
Accuracy
Recall
Precision
F-score
5
0.79 (79%)
0.782
0.7852
0.8425
10
0.834(83.4%)
0.833
0.830
0.8621
15
0.8772(87.72%)
0.892
0.8675
0.9002
20
0.9142(91.42%)
0.966
0.906
0.9504
In the results above we can see that the more we train the model the better the outcome we get when prediction the unsupervised dataset from Twitter platform. Figure below will show the loss graph to see is the mean of the model converges to 0 or approximately going to zero using the Adam optimization.
Fig 2. Loss graph of 20 epoch.
Then we also look at the accuracy of the 20 epochs to see how the mode accuracy in graph looks like.
Fig 3. Accuracy graph
The graph above shows the accuracy of the LSTM model. Next, we will focus on the prediction of the model using the unsupervised tweets from the user. The demonstration will use a table to show table is an example of how the system predict the tweets on the user.
Table IV. Prediction
Tweets
Prediction
Description
I am depressed
0.99
Negative
I am happy today
0.021
Positive
…
…
…
The system makes use of the pie graph to check to check the distribution of suicidal and non-suicidal notes. Figure 4 illustrate the distribution in pie graph.
Fig 4. Confusion matrix
The distribution of suicidal note and non-suicidal notes are respectively 82.8% and 17.2% which shows that the user is depressed and needs to get help since most of the posts they tweet are negative. This demonstrate that the system prediction can be able to classify suicidal and non-suicidal notes and produce higher outcome.
Conclusion
Text classification is the main function in the area of natural language processing. ML and DL have played a role in classification of suicidal and non-suicidal. The problem this study is trying to solve is mental health and suicide ideation assessment with social media using the DL technique. The technique used is Long Short-Term Memory (LSTM) successfully preserves the history data in length sequences, preventing gradient disappear and information loss from too much layer RNN training using memory cell and the three control gates. The gates of LSTM are the buildup layers of the neural network and update weights.
The accuracy of the training and the test dataset increases each and the epochs are increased, and the learning rate is low as possible. The result of the performance of the model is demonstrated in graphs and tables in the paper and the numbers are significant in the way the model performs. To get good results of the model need to train for longer which means more epochs need to perform as demonstrated in the paper the pie graph shows the distribution of suicidal and non-suicidal note. In the future would like to build other models to compare which model perform better than the other.
References
1. World Health Organization, 2014. Preventing suicide: A global imperative. World Health Organization.
2. ITU, “Ai for good summit,” 2019. [Online]. Available: https://aiforgood.itu.int/
3. Ke-ren, Y.U., Yun-bin, F.U. and Qi-wen, D.O.N.G., 2017. Survey on distributed word embeddings based on neural network language models. Journal of East China Normal University (Natural Science), 2017(5), p.52
4. Zirikly, A., Resnik, P., Uzuner, O. and Hollingshead, K., 2019, June. CLPsych 2019 shared task: Predicting the degree of suicide risk in Reddit posts. In Proceedings of the sixth workshop on computational linguistics and clinical psychology (pp. 24-33).
5. LeCun, Y., Bengio, Y. and Hinton, G., 2015. Deep learning. nature, 521(7553), pp.436-444.
6. Mikolov, T., Sutskever, I., Chen, K., Corrado, G.S. and Dean, J., 2013. Distributed representations of words and phrases and their compositionality. Advances in neural information processing systems, 26.
7. O'dea, B., Larsen, M.E., Batterham, P.J., Calear, A.L. and Christensen, H., 2017. A linguistic analysis of suicide-related Twitter posts. Crisis: The Journal of Crisis Intervention and Suicide Prevention, 38(5), p.319.
8. Coppersmith, G., Leary, R., Crutchley, P. and Fine, A., 2018. Natural language processing of social media as screening for suicide risk. Biomedical informatics insights, 10, p.1178222618792860.
9. Schoene, A.M., Lacey, G., Turner, A.P. and Dethlefs, N., 2019, November. Dilated lstm with attention for classification of suicide notes. In Proceedings of the tenth international workshop on health text mining and information analysis (LOUHI 2019) (pp. 136-145).
10. Hirschberg, J. and Manning, C.D., 2015. Advances in natural language processing. Science, 349(6245), pp.261-266.
11. Cambria, E. and White, B., 2014. Jumping NLP curves: A review of natural language processing research. IEEE Computational intelligence magazine, 9(2), pp.48-57.
12. Sudharsan, B., Kumar, S.P. and Dhakshinamurthy, R., 2019, December. AI Vision: Smart speaker design and implementation with object detection custom skill and advanced voice interaction capability. In 2019 11th International Conference on Advanced Computing (ICoAC) (pp. 97-102). IEEE.
13. Palaniappan, R., Sundaraj, K. and Sundaraj, S., 2014. A comparative study of the svm and k-nn machine learning algorithms for the diagnosis of respiratory pathologies using pulmonary acoustic signals. BMC bioinformatics, 15(1), pp.1-8.
14. Dzisevič, R. and Šešok, D., 2019, April. Text classification using different feature extraction approaches. In 2019 Open Conference of Electrical, Electronic and Information Sciences (eStream) (pp. 1-4). IEEE.
15. Mikolov, T., Sutskever, I., Chen, K., Corrado, G.S. and Dean, J., 2013. Distributed representations of words and phrases and their compositionality. Advances in neural information processing systems, 26.
16. Pennington, J., Socher, R. and Manning, C.D., 2014, October. Glove: Global vectors for word representation. In Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP) (pp. 1532-1543).
17. Ghannay, S., Favre, B., Esteve, Y. and Camelin, N., 2016, May. Word embedding evaluation and combination. In Proceedings of the Tenth International Conference on Language Resources and Evaluation (LREC'16) (pp. 300-305).
18. Sawhney, R., Manchanda, P., Mathur, P., Shah, R. and Singh, R., 2018, October. Exploring and learning suicidal ideation connotations on social media with deep learning. In Proceedings of the 9th workshop on computational approaches to subjectivity, sentiment and social media analysis (pp. 167-175).
19. Ji, S., Yu, C.P., Fung, S.F., Pan, S. and Long, G., 2018. Supervised learning for suicidal ideation detection in online user content. Complexity, 2018.
20. LeCun, Y., Bengio, Y. and Hinton, G., 2015. Deep learning. nature, 521(7553), pp.436-444.
21. Merkx, D., Frank, S.L. and Ernestus, M., 2022. Seeing the advantage: visually grounding word embeddings to better capture human semantic knowledge. arXiv preprint arXiv:2202.10292.
22. Zhang, J., Li, Y., Tian, J. and Li, T., 2018, October. LSTM-CNN hybrid model for text classification. In 2018 IEEE 3rd Advanced Information Technology, Electronic and Automation Control Conference (IAEAC) (pp. 1675-1680). IEEE.
