import pandas as pd 
import numpy as np

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

    def merge_clean(self):
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
        




if __name__ == "__main__":
    print(preprocess(df_bank, df_fb, df_retail).merge_clean())
