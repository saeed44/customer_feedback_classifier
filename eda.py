import preprocessing as p
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



def count_plot(df_final):
    '''
    Plot the count of each category
    '''
    fig, ax = plt.subplots(1,3,figsize=(15,4))
    sns.countplot(x=df_final.Complaint, palette="Reds_r", ax=ax[0]);
    ax[0].set_xticklabels(["Not Complaint","Complaint"])
    ax[0].set_title("Complaint Counts")

    sns.countplot(x=df_final.Suggestion, palette="Blues_r", ax=ax[1]);
    ax[1].set_xlabel("")
    ax[1].set_title("Suggestions Counts")

    sns.countplot(x=df_final.Compliment, palette="Greens_r", ax=ax[2]);
    ax[2].set_xlabel("")
    ax[2].set_title("Compliment Counts")

    fig.savefig('./results/count_per_class.png')


def dist_plot(df_final):
    '''
    Plot the distribution of the categories
    '''

    dist = pd.Series(
    {"Complaint": ((df_final["Complaint"]==1)&(df_final["Compliment"]==0)&(df_final["Suggestion"]==0)).sum(),
       "Compliment": ((df_final["Complaint"]==0)&(df_final["Compliment"]==1)&(df_final["Suggestion"]==0)).sum(),
       "Suggestion": ((df_final["Complaint"]==0)&(df_final["Compliment"]==0)&(df_final["Suggestion"]==1)).sum(),
       "Complaint-Suggestion": ((df_final["Complaint"]==1)&(df_final["Compliment"]==0)&(df_final["Suggestion"]==1)).sum(),
       "Compliment-Suggestion": ((df_final["Complaint"]==0)&(df_final["Compliment"]==1)&(df_final["Suggestion"]==1)).sum(),
       "Complaint-Compliment": ((df_final["Complaint"]==1)&(df_final["Compliment"]==1)&(df_final["Suggestion"]==0)).sum()
    } )

    sns.set_style("whitegrid")
    sns.set(font_scale = 1.4)
    fig, ax = plt.subplots(1,1,figsize=(7,7))
    sns.barplot(x=dist.index.values, y=dist.values, palette=sns.color_palette("Set2"), ax=ax,   )
    ax.set_ylabel("Count")
    plt.xticks(rotation=-45, horizontalalignment='left')
    plt.savefig('./results/class_distribution.png', bbox_inches="tight")

    
def word_count_plot(df_final):
    
    fig, ax = plt.subplots(1,1,figsize=(10,8))
    ax.set_xlabel("Word Count")
    ax.set_ylabel("Density")
    sns.kdeplot(df_final[df_final["Compliment"]==1]["word_count"],  color="tomato", ax=ax, label="Compliment" )
    sns.kdeplot(df_final[df_final["Complaint"]==1]["word_count"],  color="springgreen", ax=ax, label="Complaint" )
    sns.kdeplot(df_final[df_final["Suggestion"]==1]["word_count"],  color="cyan", ax=ax, label="Suggestion" )
    plt.legend()
    plt.savefig('./results/word_count_distribution.png')

       
       
def plot_top_n_words(series, category, n=None):
    '''
    plot top n words in a given category
    '''
    fig, ax = plt.subplots(1,1,figsize=(18,5))
    sns.set(font_scale = 1.5)
    if n:
        pass
    else:
        n=15
    sns.barplot(x=series.iloc[:n].values, y=series.iloc[:n].index.values, ax=ax)
    ax.text(.9, .5, category, transform=ax.transAxes, fontsize=17 )
    ax.set_xlabel("Word Count")
    plt.savefig(f'./results/top_n_words_{category}.png')



if __name__=="__main__":
    df_bank = pd.read_csv("./data/Banking.csv")
    df_fb = pd.read_csv("./data/FB.csv")
    df_retail = pd.read_csv("./data/Retail.csv")

    data = p.preprocess(df_bank, df_fb, df_retail)
    data.merge()
    data.clean()
    data.add_count()
    df_final = data.df_final 
    complaint_count = data.top_n_words(df_final[(df_final["Complaint"]==1)]["Text"])
    plot_top_n_words(complaint_count, "Complaint")

    compliment_count = data.top_n_words(df_final[(df_final["Compliment"]==1)]["Text"])
    plot_top_n_words(compliment_count, "Compliment")

    suggestion_count = data.top_n_words(df_final[(df_final["Suggestion"]==1)]["Text"])
    plot_top_n_words(suggestion_count, "Suggestion")
    # word_count_plot(data.df_final)

    plt.show()
