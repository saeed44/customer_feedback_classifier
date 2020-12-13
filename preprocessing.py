import pandas as pd 
import numpy as np
from nltk.corpus import stopwords
import re
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer


df_bank = pd.read_csv("./data/Banking.csv")
df_fb = pd.read_csv("./data/FB.csv")
df_retail = pd.read_csv("./data/Retail.csv")



class preprocess:
    '''
    A class to preprocess the data and prepare for training
    '''

    def __init__(self, df_bank, df_fb, df_retail):

        self.df_bank = df_bank
        self.df_fb = df_fb
        self.df_retail = df_retail
        self.df_final = None

    def merge(self):
        '''
        merge and clean the dataframes
        '''

        if "Complement" in self.df_bank.columns:
            self.df_bank.rename(columns={"Complement":"Compliment"},inplace=True)
        if "Complement" in self.df_fb.columns:
            self.df_fb.rename(columns={"Complement":"Compliment"},inplace=True)
        if "Complement" in self.df_retail.columns:
            self.df_retail.rename(columns={"Complement":"Compliment"},inplace=True) 
        if "text" in self.df_retail.columns:
               self.df_retail.rename(columns={"text":"Text"},inplace=True) 
        

        # replace Nan values with zeros (where the text does not belong to that category)
        self.df_bank.fillna(value=0, inplace=True)
        self.df_fb.fillna(value=0, inplace=True)
        self.df_retail.fillna(value=0, inplace=True) 
        

        # concatinate and randomize the dataframes
        self.df_final = pd.concat([self.df_bank, self.df_fb, self.df_retail], axis=0, ignore_index=True)
        
        # the goal is to cassify just based on text, remove other columns 
        self.df_final.drop(columns=["NPS","OSAT","NPS","touchpoint","osat","nps","campaign_name",
                                "#","TextID","campaign","TextDate",'id', 'language','date', 'nps', 
                                    'osat', 'store', 'brand','country','Sum',"SUM"], inplace=True)
        
        self.df_final = self.df_final.sample(frac=1, random_state=40).reset_index(drop=True)
        
        return self.df_final

    def clean(self):
        '''
        removes bad charachters from the text
        '''

        assert self.df_final is not None, "final dataframe is not built yet, run merge() first"

        replace_by_space = re.compile('[\n/(){}\[\]\|@,;]')
        STOPWORDS = set(stopwords.words('english'))
        BAD_SYMBOLS_RE = re.compile('[^0-9a-z !]')

        def clean(s):
            s = s.lower()
            s = replace_by_space.sub(' ',s)
            s = BAD_SYMBOLS_RE.sub('', s)
            s = ' '.join(word for word in s.split() if word not in STOPWORDS) # delete stopwors from text
            return s


        self.df_final["Text"] = self.df_final["Text"].apply(clean)

    def add_count(self):
        '''
        add the number of tokens to the final dataframe
        '''
        self.df_final["word_count"] = self.df_final["Text"].apply(lambda s: sum(Counter(s.split()).values()))
        return self.df_final


    def add_sentiments(self):
        '''
        Add sentiment scores to the dataframe
        '''
        si = SentimentIntensityAnalyzer()
        self.df_final["Sentiments"] = self.df_final["Text"].apply(lambda s: si.polarity_scores(s))
        self.df_final = pd.concat([self.df_final.drop(['Sentiments'], axis=1), self.df_final['Sentiments'].apply(pd.Series)], axis=1)    
        
        return self.df_final


    def top_n_words(self, corpus):
        '''
        The input is a corpus of words
        output is a pandas Series of word counts, descending 
        '''
        vec = CountVectorizer().fit(corpus)
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0) 
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
        
        return pd.Series({word:freq for word,freq in words_freq})
        
    def preprocess_apply(self):
        '''
        A method to do all the preprocessing, apply before training
        '''
        self.merge()
        self.clean()
        self.add_count()
        self.add_sentiments()




if __name__ == "__main__":
    data = preprocess(df_bank, df_fb, df_retail)
    # data.merge()
    # data.clean()
    # data.add_count()
    # data.add_sentiments()
    # print(data.df_final)
    # data.top_n_words(data.df_final[(data.df_final["Complaint"]==1)]["Text"])
    data.preprocess_apply()
    print(data.df_final)