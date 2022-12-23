import re
import string
import numpy as np
import pandas as pd


class Pre_Processing_data:

    def read_textfile(self, file):
        with open(file, encoding='utf8') as in_file:
            stripped = (line.strip() for line in in_file)
            tweets_ = {}
            for line in stripped:
                lines = [splits for splits in line.split("\t") if splits != ""]
                tweets_[lines[1]] = float(lines[0])
        df = pd.DataFrame(tweets_.items(), columns=['tweets', 'sentiment'])
        df = df.sample(frac=1).reset_index(drop=True)
        return df

    def cleantext(self, df, text_column=None, remove_stopwords=True, remove_punchuation=True):
        df[text_column] = df[text_column].str.lower()
        stopwords = ["a", "about", "above", "after", "again", "against",
                     "all", "am", "an", "and", "any", "are",
                     "as", "at", "be", "because",
                     "been", "before", "being", "below",
                     "between", "both", "but", "by", "could",
                     "did", "do", "does", "doing", "down", "during",
                     "each", "few", "for", "from", "further",
                     "had", "has", "have", "having", "he",
                     "he'd", "he'll", "he'stoch", "her", "here",
                     "here'stoch", "hers", "herself", "him",
                     "himself", "his", "how", "how'stoch", "i",
                     "i'd", "i'll", "i'm", "i've",
                     "if", "in", "into",
                     "is", "it", "it'stoch", "its",
                     "itself", "let'stoch", "me", "more",
                     "most", "my", "myself", "nor", "of",
                     "on", "once", "only", "or",
                     "other", "ought", "our", "ours",
                     "ourselves", "out", "over", "own", "same",
                     "she", "she'd", "she'll", "she'stoch", "should",
                     "so", "some", "such", "than", "that",
                     "that'stoch", "the", "their", "theirs", "them",
                     "themselves", "then", "there", "there'stoch",
                     "these", "they", "they'd", "they'll",
                     "they're", "they've", "this", "those",
                     "through", "to", "too", "under", "until", "up",
                     "very", "was", "we", "we'd", "we'll",
                     "we're", "we've", "were", "what",
                     "what'stoch", "when", "when'stoch",
                     "where", "where'stoch",
                     "which", "while", "who", "who'stoch",
                     "whom", "why", "why'stoch", "with",
                     "would", "you", "you'd", "you'll",
                     "you're", "you've",
                     "your", "yours", "yourself", "yourselves"]

        def remove_stopwords(data, column):
            data[f'{column} without stopwords'] = data[column].apply(
                lambda x: ' '.join([word for word in x.split() if word not in stopwords]))
            return data

        def remove_tags(string):
            result = re.sub('<*>', '', string)
            return result

        # remove html tags and brackets from text
        if remove_stopwords:
            without_stopwords = remove_stopwords(df, text_column)
            without_stopwords[f'clean_{text_column}'] = without_stopwords[
                f'{text_column} without stopwords'].apply(
                lambda cw: remove_tags(cw))
        if remove_punchuation:
            without_stopwords[f'clean_{text_column}'] = without_stopwords[f'clean_{text_column}'].str.replace(
                '[{}]'.format(string.punctuation), ' ', regex=True)

        X = without_stopwords[f'clean_{text_column}'].to_numpy()

        return X

    def sent_tokeniser(self, x):
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', x)
        sentences.pop()
        sentences_cleaned = [re.sub(r'[^\w\s]', '', x) for x in sentences]
        return sentences_cleaned

    def word_tokeniser(self, text):
        tokens = re.split(r"([-\s.,;!?])+", text)
        words = [x for x in tokens if (
                x not in '- \t\n.,;!?\\' and '\\' not in x)]
        return words

    def loadGloveModel(self, emb_path):
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

    def text_to_paragraph(self, text, para_len):
        words = text.split()
        no_paras = int(np.ceil(len(words) / para_len))
        sentences = self.sent_tokeniser(text)
        k, m = divmod(len(sentences), no_paras)
        agg_s = [sentences[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(no_paras)]
        para_s = np.array([' '.join(sents) for sents in agg_s])
        return para_s

