
# coding: utf-8


import numpy as np
import pandas as pd
import re
from sklearn.svm import LinearSVC
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer



"""
Cleans a sentence (string) by removing letters repeated more than twice, 
normalizing some letters, and removing punctuation and non Arabic letters
    
INPUT:
    text -- string, one training example from X
    
OUTPUT:
    text after having been cleaned
    
 """

def clean_tweet(text):
    try:
        #Remove elongation and repetition
        text = re.sub(r'(.)\1\1+', r'\1', text)
        
        #Normalization
        text = re.sub("[إأٱآا]", "ا", text)
        text = re.sub("ة", "ه", text)
        text = re.sub("ى", "ي", text)
        text = re.sub("ؤ", "ء", text)
        text = re.sub("ئ", "ء", text)
        
        #replace punctuation
        punctuation="[,.-_!@$%#()~`'؛.،*`\[\]:]"
        text = re.sub(punctuation, " ", text)
        
        #Remove special characters and punctuation, emojis
        not_letter="[^ ضصثقفغعهخحجكمنتالبيسشورزدءذطظ]"
        text = re.sub(not_letter, "", text)

        #remove extra whitespace
        text= re.sub('\s+',' ',text)
        text=text.strip()
    except:
        text= str(text)+" ERROR"
    return text



"""
Reads embedding files where each line begins with a word
followed by its embedding vector

INPUT:
    file -- filename where embeddings are saved
    
OUTPUT: 
    word_map -- a dictionary of embeddings, keyed by words
"""
def read_vecs(file):
    with open(file, 'r') as f:
        word_map = {}
        
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            word_map[curr_word] = np.array(line[1:], dtype=np.float64)
            
    return word_map



"""
Converts a sentence (string) into a list of words (strings). Extracts the embedding of each word
and pools its value into a single vector encoding the meaning of the sentence.
    
INPUT:
    sentence -- string, one training example from X
    word_map -- dictionary mapping every word in a vocabulary into its N-dimensional vector representation
    embedding_length -- N
    mode -- either concat or avg or min or max
    source - either aravec or senti
    
OUTPUT:
    if mode is concat:
        (Nx3)-dimensional output that has all: min, max and average pooling
        
    else if mode is avg/min/max:
        (N)-dimensional output that has specified pooling 
    
 """
def sentence_pooling(sentence, word_map, embedding_length, mode="concat", source="senti"):
     
    words = sentence.split()

    avg_v = np.zeros((embedding_length,))
    
    min_v = np.zeros((embedding_length,))
    max_v = np.zeros((embedding_length,))
   
    for i, w in enumerate(words):
        try:
            v=word_map[w]
        except KeyError:
            v=word_map["<unk>"]
                
        #Initialize first values
        if i==0:
            max_v=v
            min_v=v
            
        avg_v += v
        
        if (mode != "avg"):
            max_v=np.maximum(max_v, v)
            min_v=np.minimum(min_v, v)
            
        
    avg_v = avg_v/len(words)
    if (mode == "concat"):
        return np.concatenate([max_v, min_v, avg_v])
    elif (mode =="avg"):
        return avg_v 
    elif (mode =="min"):
        return min_v
    elif (mode =="max"):
        return max_v



"""
Takes an input a pandas dataframe with tweets cleaned in a "clean_text"
column, and converts that to its single vector encoding. Returning a 
numpy array of size (m x N) where N depends on mode, and m is the number 
of samples.

INPUT:
    df -- dataframe containing at least the "clean_text" column
    word_map -- dictionary mapping every word in a vocabulary into its N-dimensional vector representation
    embedding_length -- N
    mode -- either concat or avg
    
OUTPUT:
    if mode is concat:
        Returns a numpy array of size (m x3N)-dimensional output that has
        all: max, min and average pooling respectively
        
    else if mode is avg/min/max:
        Returns a numpy array of size (mxN)-dimensional output that has 
        the corresponding pooling only
"""


def df_to_X(df, word_map, embedding_length, mode="concat", source="senti"):
    if (mode=="concat"):
        X=np.zeros((len(df), embedding_length*3))
    else:
        X=np.zeros((len(df), embedding_length))
    i=0
    for index, row in df.iterrows():
        X[i,:]=sentence_pooling(row["clean_text"], word_map, embedding_length, mode, source)   
        i+=1
        
    return X



"""
Evaluates a liblinear SVM trained on train_X, and tested on test_X

INPUT:
    train_X -- a numpy array containing the training features of the data
    train_y -- a vector of length m1 containing the class of each training
                example, that is  0 for negative and 1 for positive
    test_X -- a numpy array containing the features of the testing data
    test_y -- a vector of length m2 containing the class of each testing
                example, that is  0 for negative and 1 for positive
    

OUTPUT:
    f1 -- f-score for each class (ordered neg, pos)
    p -- precision for each class (ordered neg, pos)
    r -- recall for each class (ordered neg, pos)

"""

def liblinear_SVM_Score(train_X, train_y, test_X, test_y):
    clf = LinearSVC(dual=False, random_state=1)
    clf.fit(train_X, train_y)
    pred_y=clf.predict(test_X)
    
    f1=f1_score(test_y, pred_y, average=None)
    p= precision_score(test_y, pred_y, average=None)
    r= recall_score(test_y, pred_y, average=None)
    return f1, p, r



"""
Trains and evaluates a dataset using on three SVM, the first taking as input 
features only, the second takes embeddings only, and the third takes in both
emeddings and features

INPUT:
    train_df -- a pandas dataframe containing the training data 
    test_df -- a pandas dataframe containing the testing data 
    word_map -- dictionary mapping every word in a vocabulary into its N-dimensional vector representation
    embedding_length -- N 
    mode -- either concat or avg or min or max
    source -- specify if it's general or sentiment embeddings, takes values 'senti' and 'aravec'
    
OUTPUT:
    None, it prints out the evaluation metrics, f-score, precision and recall
"""
def train_svm(train_df, test_df, word_map, embedding_length, mode="concat", source="senti"):
   
    
    train_X_1=train_df.loc[:, [ "tweetLength", "TweetScore", "hasHappyEmoticon", "hasSadEmoticon", 
                                     "hasQuestionMark", "hasExclamationMark", "hasIntensification", "hasPositiveWord",
                                     "hasNegativeWord","PositiveWordCount", "NegativeWordCount", "hasPositiveMPQA",
                                     "hasNegativeMPQA","hasPositiveLiu", "hasNegativeLiu"]].as_matrix() 
    test_X_1=test_df.loc[:, [ "tweetLength", "TweetScore", "hasHappyEmoticon", "hasSadEmoticon", 
                                     "hasQuestionMark", "hasExclamationMark", "hasIntensification", "hasPositiveWord",
                                     "hasNegativeWord","PositiveWordCount", "NegativeWordCount", "hasPositiveMPQA",
                                     "hasNegativeMPQA","hasPositiveLiu", "hasNegativeLiu"]].as_matrix() 
    
    train_X_2 = df_to_X(train_df, word_map, embedding_length, mode, source)
    test_X_2 = df_to_X(test_df, word_map, embedding_length, mode, source)
    
    test_y=np.array(test_df["Sentiment"]=="Positive", dtype=int) 
    train_y=np.array(train_df["Sentiment"]=="Positive", dtype=int)
    

    train_X= np.concatenate([train_X_1, train_X_2], axis=1)
    test_X= np.concatenate([test_X_1, test_X_2],axis=1)

    print ("Score with features only")
    f1, p , r= liblinear_SVM_Score(train_X_1, train_y, test_X_1, test_y)
    print ("f-score", f1, np.average(f1))
    print ("precision", p, np.average(p))
    print ("recall", r, np.average(r))
    
    print ("Score with embeddings only")
    f1e, pe , re= liblinear_SVM_Score(train_X_2, train_y, test_X_2, test_y)
    print ("f-score", f1e, np.average(f1e))
    print ("precision", pe, np.average(pe))
    print ("recall", re, np.average(re))
    
    print ("Score with features and embeddings")
    f1f, pf , rf= liblinear_SVM_Score(train_X, train_y, test_X, test_y)
    print ("f-score", f1f, np.average(f1f))
    print ("precision", pf, np.average(pf))
    print ("recall", rf, np.average(rf))
    



"""
Trains and evaluates a dataset using on two SVMs, the first taking as input 
both general and sentiment embeddings, the second takes both embeddings along with features

INPUT:
    train_df -- a pandas dataframe containing the training data 
    test_df -- a pandas dataframe containing the testing data 
    senti_embeddings -- dictionary mapping sentiment embeddings
    senti_embeddings -- dictionary mapping aravec embeddings
    embedding_length -- N 
    mode -- either concat or avg or min or max
    
OUTPUT:
    None, it prints out the evaluation metrics, f-score, precision and recall
"""
def train_svm_both(train_df, test_df, senti_embeddings, aravec, embedding_length, mode="concat"):
   
    
    train_X_1=train_df.loc[:, [ "tweetLength", "TweetScore", "hasHappyEmoticon", "hasSadEmoticon", 
                                     "hasQuestionMark", "hasExclamationMark", "hasIntensification", "hasPositiveWord",
                                     "hasNegativeWord","PositiveWordCount", "NegativeWordCount", "hasPositiveMPQA",
                                     "hasNegativeMPQA","hasPositiveLiu", "hasNegativeLiu"]].as_matrix() 
    test_X_1=test_df.loc[:, [ "tweetLength", "TweetScore", "hasHappyEmoticon", "hasSadEmoticon", 
                                     "hasQuestionMark", "hasExclamationMark", "hasIntensification", "hasPositiveWord",
                                     "hasNegativeWord","PositiveWordCount", "NegativeWordCount", "hasPositiveMPQA",
                                     "hasNegativeMPQA","hasPositiveLiu", "hasNegativeLiu"]].as_matrix() 
    
    train_X_2 = df_to_X(train_df, senti_embeddings, embedding_length, mode, "senti")
    test_X_2 = df_to_X(test_df, senti_embeddings, embedding_length, mode, "senti")
    
    train_X_3 = df_to_X(train_df, aravec, 300, mode, "aravec")
    test_X_3 = df_to_X(test_df, aravec, 300, mode, "aravec")
    
    test_y=np.array(test_df["Sentiment"]=="Positive", dtype=int) 
    train_y=np.array(train_df["Sentiment"]=="Positive", dtype=int)
    

    train_X= np.concatenate([train_X_1, train_X_2, train_X_3], axis=1)
    test_X= np.concatenate([test_X_1, test_X_2, test_X_3],axis=1)
    
    print ("Score with both embeddings")
    f1, p , r= liblinear_SVM_Score(np.concatenate([train_X_2, train_X_3], axis=1), train_y, np.concatenate([test_X_2, test_X_3], axis=1), test_y)
    print ("f-score", f1, np.average(f1))
    print ("precision", p, np.average(p))
    print ("recall", r, np.average(r))
    
    print ("Score with features and both embeddings")
    f1f, pf , rf= liblinear_SVM_Score(train_X, train_y, test_X, test_y)
    print ("f-score", f1f, np.average(f1f))
    print ("precision", pf, np.average(pf))
    print ("recall", rf, np.average(rf))
    



"""
Trains and evaluates a dataset using a baseline SVM, either by presence or 
count bag of words

INPUT:
    train_df -- a pandas dataframe containing the training data in at least 
                two columns, "clean_text" and "Sentiment"
    test_df -- a pandas dataframe containing the testing data in at least 
                two columns, "clean_text" and "Sentiment"
    mode -- either count or presence
    
OUTPUT:
    None, it prints out the evaluation metrics, f-score, precision and recall
"""
def train_baseline_SVM(train_df, test_df, mode='count'):
    if (mode=="count"):
        vectorizer = CountVectorizer()
    else:
        vectorizer = CountVectorizer(binary=True)
        
    corpus = pd.concat([train_df["clean_text"], test_df["clean_text"]])

    test_y=np.array(test_df["Sentiment"]=="Positive", dtype=int)
    train_y=np.array(train_df["Sentiment"]=="Positive", dtype=int)

    X = vectorizer.fit_transform(corpus)
    m=train_df.shape[0]
    
    train_X=X.toarray()[:m,:]
    test_X=X.toarray()[m:,:]

    f1, p , r= liblinear_SVM_Score(train_X, train_y, test_X, test_y)
    print ("f-score", f1, np.average(f1))
    print ("precision", p, np.average(p))
    print ("recall", r, np.average(r))


