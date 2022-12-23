import pandas as pd

from Feature_Extraction.pre_processing import Pre_Processing_data

txt_proc = Pre_Processing_data()
path = '../../Project/Research/Mental_Health_and_Sucide_Ideation_assement_with_social_media/Dataset/data/tweets_train.txt'
train_df = txt_proc.read_textfile(path)
X_train = txt_proc.cleantext(train_df, text_column='tweets', remove_stopwords=True, remove_punchuation=True)[0:2000]
Y_train = train_df['sentiment'].to_numpy()[0:3250]

test_df = txt_proc.read_textfile('Dataset/data/tweets_test.txt')
X_test = txt_proc.cleantext(test_df, text_column='tweets', remove_stopwords=True, remove_punchuation=True)[0:2000]
Y_test = test_df['sentiment'].to_numpy()[0:3250]


tweets_data = 'Dataset/uploaded/prediction.csv'
tweets_df = pd.read_csv(tweets_data)
X_pred = txt_proc.cleantext(tweets_df, text_column='timeline', remove_stopwords=True, remove_punchuation=False)
tweets_line = tweets_df['month'].to_numpy()

glove_file = '../../Project/Research/Mental_Health_and_Sucide_Ideation_assement_with_social_media/Dataset/word_embedded/glove.6B.300d.txt'
emb_matrix = txt_proc.loadGloveModel(glove_file)