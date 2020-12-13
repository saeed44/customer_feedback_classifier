import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.compose import make_column_selector
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegressionCV
from joblib import dump, load
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import preprocessing as p




class Model:

    def __init__(self, df_final):
        
        self.df_final = df_final

    
    def split(self):
        '''
        Train test split
        '''
        X = self.df_final[['Text','neg','neu','pos','compound','word_count']]
        y = self.df_final[['Compliment','Complaint','Suggestion']]
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state = 62, stratify=y['Complaint'])

        col_trans = ColumnTransformer(
            [('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1,1)), 'Text'),
            ('neg', StandardScaler(),make_column_selector(pattern='neg',dtype_include=np.number)),
            ('neu', StandardScaler(),make_column_selector(pattern='neu',dtype_include=np.number)),
            ('pos', StandardScaler(),make_column_selector(pattern='pos',dtype_include=np.number)),
            ('compound', StandardScaler(),make_column_selector(pattern='compound',dtype_include=np.number)),
    #      ('Service_Banking', StandardScaler(),make_column_selector(pattern='Service_Banking',dtype_include=np.number)),
    #      ('Service_FB', StandardScaler(),make_column_selector(pattern='Service_FB',dtype_include=np.number)),
    #      ('Service_Retail', StandardScaler(),make_column_selector(pattern='Service_Retail',dtype_include=np.number)),
            ('word_count', StandardScaler(),make_column_selector(pattern='word_count',dtype_include=np.number))] ,
            remainder='drop')

        X_train_f = col_trans.fit_transform(X_train)
        X_test_f = col_trans.transform(X_test)      
    
        return X_train_f, X_test_f, y_train, y_test

    def train(self):
        '''
        Train the model
        '''
        X_train, X_test, y_train, y_test = self.split()

        log_reg = MultiOutputClassifier(LogisticRegressionCV(cv=5, random_state=21,  class_weight='balanced', scoring="f1",max_iter=500 )) # LogisticRegressionCV(cv=5, random_state=0)
        log_reg.fit(X_train, y_train)

        dump(log_reg, './results/model_log_reg.joblib') 


    def test(self):
        
        '''
        Test the model, print metrics and plot and save confusion matrix
        '''

        X_train, X_test, y_train, y_test = self.split()

        log_reg = load('./results/model_log_reg.joblib')

        y_pred = log_reg.predict(X_test)
        y_pred = pd.DataFrame({'Compliment': y_pred[:,0], 'Complaint': y_pred[:,1], 'Suggestion': y_pred[:,2] })

        print("Compliment:","\n",classification_report(y_test['Compliment'], y_pred['Compliment'] ))
        print("Complaint:","\n",classification_report(y_test['Complaint'], y_pred['Complaint']))
        print("Suggestion:","\n",classification_report(y_test['Suggestion'], y_pred['Suggestion']))

        fig, ax = plt.subplots(1,3,figsize=(11,2))
        sns.heatmap(confusion_matrix(y_test["Compliment"],y_pred["Compliment"]),annot=True, fmt="g",  cmap="Blues", ax=ax[0], cbar=False)
        sns.heatmap(confusion_matrix(y_test["Complaint"],y_pred["Complaint"]) , annot=True, fmt="g",  cmap="Blues", ax=ax[1], cbar=False)
        sns.heatmap(confusion_matrix(y_test["Suggestion"],y_pred["Suggestion"]) ,annot=True, fmt="g",  cmap="Blues", ax=ax[2])       
        ax[0].set_title("Compliment")
        ax[1].set_title("Complaint")
        ax[2].set_title("Suggestion")
        for i in range(3):
            ax[i].set_ylabel("True Label")
            ax[i].set_xlabel("Predicted Label")

        plt.savefig(f'./results/confusion_matrix.png')
        plt.show()



if __name__ == "__main__":
    df_bank = pd.read_csv("./data/Banking.csv")
    df_fb = pd.read_csv("./data/FB.csv")
    df_retail = pd.read_csv("./data/Retail.csv")

    data = p.preprocess(df_bank, df_fb, df_retail)
    data.preprocess_apply()
    df_final = data.df_final
    tr = Model(df_final)

    tr.test()