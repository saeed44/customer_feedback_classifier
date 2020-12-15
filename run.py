import pandas as pd
import os
import sys
import eda
from models import Model
import preprocessing as p

process = sys.argv[1]
path1 = sys.argv[2]
path2 = sys.argv[3]
path3 = sys.argv[4]
    
def read_files():

    if os.path.exists(path1) and os.path.exists(path2) and os.path.exists(path3):  
        df_1 = pd.read_csv(path1)
        df_2 = pd.read_csv(path2)
        df_3 = pd.read_csv(path3)
        return df_1, df_2, df_3
    else:
        raise FileExistsError("File(s) does not exist.")

def run(process):
    '''
    This runs the given process
    '''
    df_1, df_2, df_3 = read_files()
    
    data = p.preprocess(df_1, df_2, df_3)
    data.merge()
    data.clean()
    data.add_count()
    data.add_sentiments()
    df_final = data.df_final 

    if process=="eda":
        eda.run_eda(df_1, df_2, df_3)
    elif process=="train":
        m = Model(df_final)
        m.train()
    elif process=="test":
        m = Model(df_final)
        m.test()   



if __name__=="__main__":

    print(process)
    run(process)