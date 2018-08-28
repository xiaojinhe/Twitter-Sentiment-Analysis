import re
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

class PreprocessingConfig(object):
    replace_abbrevation_op = True
    clean_hashtag_op = True
    clean_url_op = True
    clean_mention_tag_op = True
    clean_reserved_words_op = False
    translate_emoji_op = True
    correct_spelling_op = True

class DataPreprocessing(object):

    def __init__(self, config, raw_data_file, cleaned_data_dir):
        self.config = config
        if self.config.translate_emoji_op:
            self.positive, self.negative, self.neutral = self.build_emoji_dict()

        if self.config.correct_spelling_op:
            self.dict = self.build_corpus_dict()

        self.data_clean_tweet(raw_data_file, cleaned_data_dir)

    def replace_abbreviation(self, str):

        cleaned_str = re.sub(r"\'ve", "have", str) # I've => I 've
        cleaned_str = re.sub(r"\'s", " \'s", cleaned_str) # It's = > it 's
        cleaned_str = re.sub(r"won\'t", "will not", cleaned_str)
        cleaned_str = re.sub(r"n\'t", " not", cleaned_str) # aren't => are not
        cleaned_str = re.sub(r"\'d", " would", cleaned_str) # I'd => I 'd
        cleaned_str = re.sub(r"\'re", " are", cleaned_str) # they're => they 're
        cleaned_str = re.sub(r"\'ll", " will", cleaned_str) # I'll => I 'll
        cleaned_str = re.sub(r"i\'m", "i am", cleaned_str) 
        cleaned_str = re.sub(r",", " , ", cleaned_str)
        cleaned_str = re.sub(r"!", " ! ", cleaned_str)
        cleaned_str = re.sub(r"\(", " \( ", cleaned_str)
        cleaned_str = re.sub(r"\)", " \) ", cleaned_str)
        cleaned_str = re.sub(r"\?", " \? ", cleaned_str)
        cleaned_str = re.sub(r"\s{2,}", " ", cleaned_str) # delete addtional whitespaces
        return cleaned_str

    def remove_hashtag(self, str):
        HASHTAG_PATTERN = re.compile(r'#\w*')
            # remove hashtag
        cleaned_str = re.sub(HASHTAG_PATTERN, "", str)
        return cleaned_str

    def remove_url(self, str):
        URL_PATTERN = re.compile(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?\xab\xbb\u201c\u201d\u2018\u2019]))')
        cleaned_str = re.sub(URL_PATTERN, "", str)
        return cleaned_str

    def remove_mention_tag(self, str):
        MENTION_PATTERN = re.compile(r'@\w*')
        cleaned_str = re.sub(MENTION_PATTERN, "", str)
        return cleaned_str

    def remove_reserved_words(self, str):
        RESERVED_WORDS_PATTERN = re.compile(r'^(RT|FAV)')  
        cleaned_str = re.sub(RESERVED_WORDS_PATTERN, "", str)
        return cleaned_str  

    def build_emoji_dict(self):
        positive = []
        negative =[]
        neutral = []

        eyes = ["8",":","=",";"]
        noses = ["'","`","-",r"\\"]

        for eye in eyes:
            for nose in noses:
            # from left to right: eye, nose, mouth
                for mouth in ["\)", "]", "}", "d", "p"]:
                    positive.append(eye+nose+mouth)
                    positive.append(eye+mouth)
                for mouth in ["\(", "\[", "{"]: 
                    negative.append(eye+nose+mouth)
                    negative.append(eye+mouth)
                for mouth in ["\|", "\/", r"\\"]: 
                    neutral.append(eye+nose+mouth)
                    neutral.append(eye+mouth)

            # from left to right: mouth, nose, eye
                for mouth in ["\(", "\[", "{", "d", "p"]:
                    positive.append(mouth+nose+eye)
                    positive.append(mouth+eye)
                for mouth in ["\)", "]", "}"]: 
                    negative.append(mouth+nose+eye)
                    negative.append(mouth+eye)
                for mouth in ["\|", "\/", r"\\"]: 
                    neutral.append(mouth+nose+eye)
                    neutral.append(mouth+eye)

        return list(set(positive)), list(set(negative)), list(set(neutral))

    def translate_emoji(self, str):
        hearts = ["<3", "♥"]
        translated_words = []
        for word in str.split():
            if word in hearts:
                translated_words.append("<love>")
            elif word in self.positive:
                translated_words.append("<happy>")
            elif word in self.negative:
                translated_words.append("<sad>")
            elif word in self.neutral:
                translated_words.append("<neutral>")
            else:
                translated_words.append(word)
        return " ".join(translated_words)

    def build_corpus_dict(self):
        """ Build a dict using three normalization corpus to correct common typo and abbreviations. 
        Links:
        http://luululu.com/tweet/typo-corpus-r1.txt
        http://people.eng.unimelb.edu.au/tbaldwin/etc/emnlp2012-lexnorm.tgz
        http://www.hlt.utdallas.edu/~yangl/data/Text_Norm_Data_Release_Fei_Liu/
        """
        dict = {}
        dict1 = open("./twitter-sentiment/data/tweet_typo_corpus.txt", 'rb')
        for word in dict1:
            word = word.decode('utf-8').split()
            dict[word[0]] = word[1]
        dict1.close()
        dict2 = open("./twitter-sentiment/data/tweet_typo_corpus2.txt", 'rb')
        for word in dict2:
            word = word.decode('utf-8').split()
            dict[word[0]] = word[1]
        dict2.close()
        dict3 = open("./twitter-sentiment/data/tweet_corpus.txt", 'rb')
        for word in dict3:
            word = word.decode('utf-8').split()
            dict[word[1]] = word[3]
        dict3.close()
        return dict

    def correct_spelling(self, str):
        correct_str = []
        for word in str.split():
            if word in self.dict.keys():
                correct_str.append(self.dict[word])
            else:
                correct_str.append(word)
        return ' '.join(correct_str)


    def clean_text(self, str):
        """ String cleaning for dataset """
        soup = BeautifulSoup(str, 'lxml')
        souped = soup.get_text()
        try:
            cleaned_str = souped.decode("utf-8-sig").replace(u"\ufffd", "?")
        except:
            cleaned_str = souped
        cleaned_str = str
        if self.config.clean_url_op:
            cleaned_str = self.remove_url(cleaned_str)
        if self.config.clean_hashtag_op:
            cleaned_str = self.remove_hashtag(cleaned_str)
        if self.config.clean_mention_tag_op:
            cleaned_str = self.remove_mention_tag(cleaned_str)
        if self.config.clean_reserved_words_op:
            cleaned_str = self.remove_reserved_words(cleaned_str)

        cleaned_str = cleaned_str.lower()
        if self.config.replace_abbrevation_op:
            cleaned_str = self.replace_abbreviation(cleaned_str)
    
        if self.config.translate_emoji_op:
            cleaned_str = self.translate_emoji(cleaned_str)

        if self.config.correct_spelling_op:
            cleaned_str = self.correct_spelling(cleaned_str)

        cleaned_str = re.sub(r"[^a-z(),!?\'\`]", " ", cleaned_str)
        cleaned_str = re.sub(r"\s{2,}", " ", cleaned_str) # delete addtional

        return cleaned_str.strip()

    def data_clean_tweet(self, raw_data_file, cleaned_data_dir):
        column_names = ['sentiment', 'id', 'date', 'query_string', 'user', 'text']
        df = pd.read_csv(raw_data_file, header=None, names=column_names, encoding = "ISO-8859-1")
        df.drop(['id', 'date', 'query_string', 'user'], axis=1, inplace=True)
        df['sentiment'] = df['sentiment'].map({0:0, 4:1})

        print("\nStart cleaning data......")
        cleaned_text = []
        lenOfdf = len(df)
        for i in range(0, lenOfdf):
            if ((i + 1) % 100000 == 0):
                print("%d of %d texts has been cleaned." % (i + 1, lenOfdf))
            cleaned_text.append(self.clean_text(df['text'][i]))

        cleaned_df = pd.DataFrame(cleaned_text, columns=['text'])
        cleaned_df['sentiment'] = df.sentiment
        cleaned_df['text'].replace("", np.nan, inplace=True)
        cleaned_df.dropna(inplace=True)
        cleaned_df.reset_index(drop=True, inplace=True)
        print(cleaned_df.info())

        pos_tweets = cleaned_df[cleaned_df.sentiment == 1]
        pos_tweets.reset_index(drop=True, inplace=True)
        print(pos_tweets.info())
        neg_tweets = cleaned_df[cleaned_df.sentiment == 0]
        neg_tweets.reset_index(drop=True, inplace=True)
        print(neg_tweets.info())

        test_percentage = 0.1
        # training set
        train_pos_set = pos_tweets.iloc[0:int(len(pos_tweets) * (1 - test_percentage))]
        train_neg_set = neg_tweets.iloc[0:int(len(neg_tweets) * (1 - test_percentage))]
        train_pos_set.to_csv(cleaned_data_dir + "/" + "cleaned_pos_file.csv", encoding="utf-8")
        train_neg_set.to_csv(cleaned_data_dir + "/" + "cleaned_neg_file.csv", encoding="utf-8")

        test_pos_set = pos_tweets.iloc[len(train_pos_set):]
        test_neg_set = neg_tweets.iloc[len(train_neg_set):]
        test_pos_set.to_csv(cleaned_data_dir + "/" + "cleaned_pos_test.csv", encoding="utf-8")
        test_neg_set.to_csv(cleaned_data_dir + "/" + "cleaned_neg_test.csv", encoding="utf-8")

        """small_pos_tweets = pos_tweets.iloc[0:30000]
        small_pos_tweets.reset_index(drop=True, inplace=True)
        small_neg_tweets = neg_tweets.iloc[0:30000]
        small_neg_tweets.reset_index(drop=True, inplace=True)
        small_pos_tweets.to_csv(cleaned_data_dir + "/" + "small_pos_file.csv", encoding="utf-8")
        small_neg_tweets.to_csv(cleaned_data_dir + "/" + "small_neg_file.csv", encoding="utf-8")"""

if __name__ == '__main__':
    raw_data_file = "./twitter-sentiment/data/raw_data_file.csv"
    cleaned_data_dir = "./twitter-sentiment/data/"
    config = PreprocessingConfig()
    preprocessor = DataPreprocessing(config, raw_data_file, cleaned_data_dir)





