import pandas as pd
import numpy as np
import os, string
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
load_dotenv()

SHEET_ID = os.getenv("SHEET_ID")
CLASS_COLUMN = os.getenv("CLASS_COLUMN")

def read_response(url, CLASS_ID):    
    """
    This function reads the results of the GSheet as a result 
    of filling out the background check's GForm.
    """
    df = pd.read_csv(url)
    df = df[df[CLASS_COLUMN].str.endswith(CLASS_ID)]
    df.drop(CLASS_COLUMN, axis=1, inplace=True)
    df.reset_index(inplace=True, drop=True)
    df.sample(5)
    return df

def simple_clean(txt):
    """
    This function does: 
      - lowering all letters
      - removing unnecessary space
      - removing punctuation.
    """
    if len(str(txt))<1: return txt
    txt = txt.lower()
    txt = txt.strip()
    txt = txt.translate(str.maketrans("","",string.punctuation))
    return txt

def viz_summary_1(df):
    """
    This function will create a bar plot as a "Background Check" result summary
    """
    cols = df.columns

    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    plt.rcParams['axes.labelsize'] = 15

    plt.style.use("seaborn-darkgrid")
    plt.figure(figsize=(16, 8))
    plt.subplots_adjust(hspace=.8, wspace=.6)
    plt.suptitle(f"Response Summary (last update: {df.iloc[-1][df.columns[0]]})",
                fontsize=20, fontweight="bold")

    for i, col in enumerate(cols[1:5]):
        plt.subplot(2, 2, i+1)
        ax = sns.countplot(data=df, x=cols[i + 1], order=df[cols[i + 1]].value_counts().index)

        total = len(df[cols[i + 1]])
        for p in ax.patches:
            percentage = p.get_height() / total * 100
            ax.annotate(f'{percentage:.1f}%', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 5), textcoords='offset points',
                        fontsize=12.5)

    plt.tight_layout()
    plt.show()

def preprocess_text_data(df):
    """
    This function aims to process text data from the "Background Check" Form 
    before it is visualized.
    """
    cols = df.columns
    df_txt = df[cols[-4:-1]].copy()
    df_txt.replace({np.nan:"", "-":"", "None":""}, inplace=True)

    for col in df_txt.columns[1:]:
        df_txt[col] = df_txt[col].apply(lambda t: simple_clean(t))

    df_tmp = df_txt.copy()
    df_tmp.replace('', np.nan, inplace=True)
    df_tmp.dropna(inplace=True)
    df_tmp["L"] = df_tmp[df_tmp.columns[-1]].apply(lambda l:len(str(l)))
    df_tmp.sort_values([df_tmp.columns[0], df_tmp.columns[-1]], ascending=[False]*2, inplace=True)
    df_tmp.drop("L", axis=1, inplace=True)
    return df_tmp

def viz_summary_2(df_tmp, col_idx):
    """
    This function is to visualize the results of the "Background Check" form
    in the form of a narrative.
    """
    sentences = df_tmp[df_tmp.columns[col_idx]].copy()
    col_dict = {5:"blue", 4:"orange", 3:"green", 2:"red"}
    df_tmp["color"] = df_tmp[df_tmp.columns[0]].apply(lambda l:col_dict[l])
    colors = df_tmp[df_tmp.columns[-1]].copy()
    df_tmp.drop("color", axis=1, inplace=True)
    text = ' '.join(sentences)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    for i, (sentence, color) in enumerate(zip(sentences,colors)):
        x = 5
        y = 8 - i 
        ax.text(x, y, sentence, fontsize=16, ha='center', va='center', color=color)

    plt.title(f"{df_tmp.columns[col_idx]}", fontsize=18)
    plt.show()