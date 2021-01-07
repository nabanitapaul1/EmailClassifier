#!/usr/bin/env python
# coding: utf-8

# ## Business Objective
# 
#  
# 
# #### To Classify abusive emails and non-abusive emails
# 
# ###### Author: Nabanita Paul
# 

# In[135]:


# import libraries

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Dataset details

# In[136]:


# Import data
#email_data =  pd.read_excel("D:\\Project\\EmailClassification\\emails.xlsx")
email_data =pd.read_csv("D:\\Project\\EmailClassification\\emails.csv")


# In[137]:


email_data.head()


# In[138]:


email_data.shape


# In[139]:


email_data.dtypes


# In[140]:


email_data.info()


# In[141]:


email_data.columns


# In[142]:


email_data ["Class"].value_counts().head()


# #### Columns to be considered
# 
# #### 1. Class: Target Variable  that has to be predicted - Abusive, Non Abusive 
# #### 2. Content: This varable contains the text of the email. This is the main column on which NLP techniques will be used.
# 

# In[143]:


email_data_ab = email_data[["content","Class"]]


# In[144]:


email_data_ab.head()


# In[145]:


# filtereing only abusive and non-abusive data
#email_data_ab= email_data[(email_data["Class"]=="Abusive") | (email_data["Class"]=="Non Abusive")]
#type(email_data_ab)


# # Exploratory Data Analysis

# In[146]:


print(email_data_ab["Class"].value_counts())
sns.countplot(email_data_ab["Class"])


# In[147]:


email_data_ab.groupby('Class').describe()


# ## Added a variable content_count ---- Count the length of the content for each text content

# In[148]:



email_data_ab['content_count']=email_data_ab['content'].apply(lambda x: len(str(x)))
email_data_ab.head()


# In[149]:


email_data_ab['content_count'].describe()


# In[150]:


# Longest mail

email_data_ab[email_data_ab['content_count']==272036]['content'].iloc[0]


# In[151]:


# Shortest mail
email_data_ab[email_data_ab['content_count']==1]['content'].iloc[0]


# In[152]:


# Importing NLP libraries

# pip install nltk


# In[153]:


#nltk.download()


# ## Data Preprocessing
# 
# 

# ## A. Data Cleaning

# In[154]:


import re


# #### 1.Removal of "n\" characters

# In[155]:


email_data_ab["content_w_space"]=email_data_ab["content"].replace('\n'," ",regex=True)


# In[156]:


email_data_ab["content_w_space"].head(10)


# In[157]:


def to_lower(text):
    result = str(text).lower()
    return result


# In[158]:


email_data_ab["content_low"]=email_data_ab["content_w_space"].apply(lambda x: to_lower(x))


# In[159]:


email_data_ab["content_low"].head()


# #### 3. Removal of special characters and numbers

# In[160]:


# it removes special character and numbers 

def remove_special_characters(text):
    #result = re.sub("[^A-Za-z0-9]+"," ", text)
    result =  re.sub(r'[^a-zA-Z]', ' ', text)
    return result


# In[161]:


email_data_ab["content_wsch"]=email_data_ab["content_low"].apply(lambda x: remove_special_characters(x))


# In[162]:


email_data_ab["content_wsch"].head()


# #### 4. Removal of unicode characters
# 

# In[163]:


#def removal_unicode(text):
 #   text = re.sub(r'[^\\x00-\\x7F]+', '', text)
 #   return text


# In[164]:


#email_data_ab.loc["content_wunicode"] = email_data_ab["content_wsch"].apply(lambda x: removal_unicode(x))


# #### 5. Removal of hyperlinks

# In[165]:


def removal_hyperlinks(text):
    result =  re.sub(r"http\\S+", " ", str(text))
    return result


# In[166]:


email_data_ab["content_whl"]=email_data_ab["content_wsch"].apply(lambda x: removal_hyperlinks(x))


# In[167]:


email_data_ab["content_whl"].head()


# #### 6. Removal of whitespaces

# In[168]:


def removal_whitespaces(text):
    result =  re.sub(' +', ' ', text)
    return result


# In[169]:


email_data_ab["content_wws"]=email_data_ab["content_whl"].apply(lambda x: removal_whitespaces(x))


# In[170]:


email_data_ab["content_wws"].head()


# #### 7. Removal of stopwords

# In[171]:


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
#print(stopwords.words('english'))
stop_words = set(stopwords.words('english'))


# In[172]:


def removal_stopwords(text):
    word_tokens = word_tokenize(text)  
    filtered_sentence = []
    a_row=""
    for a_word in word_tokens:
        if a_word not in stop_words:
            filtered_sentence.append(a_word)
            a_row = " ".join(filtered_sentence)
    return a_row


# In[173]:


email_data_ab["content_w_sw"]=email_data_ab.content_wws.apply(lambda x: removal_stopwords(x))


# In[174]:


email_data_ab["content_w_sw"].head()


# ## Text normalization

# #### 1.Stemming

# In[175]:


from nltk.stem import PorterStemmer


# In[176]:


stemer = PorterStemmer()
def stemming(text):
    word_tokens = word_tokenize(text)  
    a_array=[]
    a_string = ""
    for a_word in word_tokens:
   
       a_stem = stemer.stem(a_word)
   
       a_array.append(a_stem)
    
       a_string = " ".join(a_array)
    return a_string


# In[177]:


#email_data_ab["content_stem"]=email_data_ab.content_w_sw.apply(lambda x: stemming(x))


# In[178]:


#email_data_ab["content_stem"].head()


# #### 2. Lemmatization
# 

# In[179]:


from nltk import WordNetLemmatizer


# In[180]:


lemma = WordNetLemmatizer()
def lemmatization(text):
    word_tokens = word_tokenize(text) 
    a_array=[]
    a_string = ""
    for a_word in word_tokens:
               
        a_lemma = lemma.lemmatize(a_word,pos = "n")
        a_lemma1 = lemma.lemmatize(a_lemma, pos="v")
        a_lemma2 = lemma.lemmatize(a_lemma1, pos="a")
   
        a_array.append(a_lemma2)
        
        a_string = " ".join(a_array)
    return a_string


# In[181]:


email_data_ab["content_lemma"]=email_data_ab.content_w_sw.apply(lambda x: lemmatization(x))


# In[182]:


email_data_ab["content_lemma"].head()


# In[183]:


email_data_ab["content_cleaned"]=email_data_ab["content_lemma"]


# ####  Newly count content

# In[184]:


email_data_ab["length"] = email_data_ab["content_cleaned"].apply(lambda x: len(x))


# In[185]:


email_data_ab["length"].describe()


# In[186]:


email_data_final = email_data_ab[["content_cleaned","Class","length"]]


# In[187]:


email_data_final.head()


# In[188]:


len(email_data_final)


# #### Checking for duplicates

# In[189]:


duplicate_records = email_data_final[email_data_final.duplicated()] 
duplicate_records.head(5)


# In[190]:


len(duplicate_records)


# #### There are 25314 duplicated values.

# #### Drop duplicates
# 

# In[191]:



email_data_final =  email_data_final.drop_duplicates() # keeping the first value
email_data_final.head()


# In[192]:


email_data_final.shape


# In[193]:


email_data_final.to_csv("email_data_final.csv")


# ## Visualization and Word Cloud

# In[194]:


# Target Variable Class
    
print(email_data_final["Class"].value_counts())
sns.countplot("Class", data = email_data_final)
                       


# In[195]:


email_data_final['length'].plot(bins=50,kind='hist')


# In[196]:


# email_data_final.to_csv("email_data_final.csv")


# In[197]:


email_data_final['length'].describe()


# In[198]:


# Longest mail

email_data_final[email_data_final['length']==253499]['content_cleaned'].iloc[0]


# In[199]:


# Shortest mail

email_data_final[email_data_final['length']==0]['content_cleaned'].iloc[0]


# #### Filtering Abusive mails

# In[200]:



email_abusive= email_data_final[(email_data_final["Class"]=="Abusive")]
email_abusive.shape


# In[201]:


email_abusive.content_cleaned[0:5]


# #### Filtering Non Abusive mail
# 

# In[202]:



email_non_abusive= email_data_final[(email_data_final["Class"]=="Non Abusive")]
email_non_abusive.shape


# #### Abusive Mails
# 

# #### Building Text Corpus

# In[203]:


final_email_abusive=""
abusive_email =[]
for text in email_abusive["content_cleaned"]:
    abusive_email.append(text)
    final_email_abusive =  "".join(abusive_email)
final_email_abusive


# #### Word Clouds for abusive emails
# 

# In[204]:


from wordcloud import WordCloud,STOPWORDS


# In[205]:


stopwords = set(STOPWORDS) 

wordcloud_abusive_words = WordCloud(
        background_color='white',
        height = 4000,
        width=4000,
        stopwords = stopwords,
        min_font_size = 10
   ).generate(final_email_abusive)


# In[206]:


#plt.figure(figsize = (40,40))
plt.figure(figsize = (8, 8), facecolor = None) 
plt.axis("off") 
plt.tight_layout(pad = 0)  
plt.imshow(wordcloud_abusive_words,interpolation="bilinear")


# #### Non Abusive

# #### Building Text Corpus

# In[207]:


final_email_nonabusive=""
nonabusive_email =[]
for text in email_non_abusive["content_cleaned"]:
    nonabusive_email.append(text)
    final_email_nonabusive =  "".join(nonabusive_email)
final_email_nonabusive


# In[208]:


wordcloud_nonabusive_words = WordCloud(
        background_color='white',
        height = 4000,
        width=4000,
        stopwords = stopwords,
        min_font_size = 10
   ).generate(final_email_nonabusive)


# In[209]:


#plt.figure(figsize = (40,40))
plt.figure(figsize = (8, 8), facecolor = None) 
plt.axis("off") 
plt.tight_layout(pad = 0)  
plt.imshow(wordcloud_nonabusive_words,interpolation="bilinear")


# In[210]:


#### N Gram Visualization


# In[211]:


#headlines = prof_headlines_na.value_counts().head(20)
#plt.figure(figsize=(12,8))
#sns.barplot(headlines.values, headlines.index)
#plt.xlabel("Count", fontsize=15)
#plt.ylabel("Unique Headlines", fontsize=15)
#plt.title("Top 20 Unique Professionals Headlines")
#plt.show()


# ##  Tokenization

# In[212]:


email_data_final.columns


# In[213]:


from nltk.tokenize import word_tokenize


# In[214]:


email_data_final["content_tokenized"]= email_data_final["content_cleaned"].apply(lambda x: word_tokenize(x) )


# In[215]:


email_data_final["content_tokenized"].head()


# In[216]:


email_data_final.columns


# ## Bag of words/ Vectorization

# In[217]:


from sklearn.feature_extraction.text import CountVectorizer


# In[218]:


cv = CountVectorizer()


# In[219]:


X1=cv.fit(email_data_final["content_cleaned"])


# In[220]:


#X=cv.fit_transform(email_data_final["content_cleaned"])


# In[221]:


#X_feature= X.toarray()


# In[222]:


#X_feature


# In[223]:


#X_feature.shape


# In[224]:


print(len(X1.vocabulary_))


# ####  There are 22762 documents and 99784 unique words

# #### Analyzing one of the email

# In[225]:



a_email=email_data_final['content_cleaned'][4]
a_email


# In[226]:


a_email_vector=X1.transform([a_email])
print(a_email_vector)
print(a_email_vector.shape)


# #### This means that there are nineteen unique words in 5th email   (after removing common stop words). 
# #### one of them appear twice ,one of them appear thrice and the rest only once. Let's go ahead and check and confirm which ones appear twice and trice:

# In[227]:


print(X1.get_feature_names()[24004])
print(X1.get_feature_names()[43987])


# In[228]:


emails= X1.transform(email_data_final['content_cleaned']) # Transformig the entire corpus


# In[229]:


emails.shape


# In[230]:


print('Shape of Sparse Matrix: ',emails.shape)
print('Amount of non-zero occurences:',emails.nnz)


# In[231]:


#sparsity =(100.0 * emails.nnz/(emails.shape[0]*emails.shape[1]))
#print('sparsity:{}'.format(round(sparsity)))


# #### Term-frequency- Inverse Document frequency

# In[232]:


from sklearn.feature_extraction.text import TfidfTransformer


# In[233]:


tfidf_transformer=TfidfTransformer().fit(emails)


# In[234]:


# For 5th emails

tfidf_a_email = tfidf_transformer.transform(a_email_vector)
print(tfidf_a_email.shape)


# In[235]:


print(tfidf_transformer.idf_[X1.vocabulary_['gamble']])
print(tfidf_transformer.idf_[X1.vocabulary_['asshole']])
print(tfidf_transformer.idf_[X1.vocabulary_['excelr']])
print(tfidf_transformer.idf_[X1.vocabulary_['ect']])
print(tfidf_transformer.idf_[X1.vocabulary_['make']])
print(tfidf_transformer.idf_[X1.vocabulary_['lavorato']])
print(tfidf_transformer.idf_[X1.vocabulary_['problem']])
print(tfidf_transformer.idf_[X1.vocabulary_['go']])


# In[236]:


#'asshole john j lavorato excelr john arnold hou ect ect cc subject john cant seem make gamble problem go away bill denver jack'


# In[237]:


emails_tfidf = tfidf_transformer.transform(emails) # transforming the entire corpus
print(emails_tfidf.shape)


# In[238]:


emails_tfidf.shape


# ## Label Encoding 

# In[239]:


# Import label encoder 
from sklearn import preprocessing 
  
# label_encoder object knows how to understand word labels. 
label_encoder = preprocessing.LabelEncoder() 
  
# Encode labels in column 'species'. 
email_data_final['Class']= label_encoder.fit_transform(email_data_final['Class']) 
  
email_data_final['Class'].unique()


# In[240]:


email_data_final["Class"].value_counts()


# ## Training a Model 

# In[241]:


from sklearn.naive_bayes import MultinomialNB
abusive_detect_model = MultinomialNB().fit(emails_tfidf,email_data_final['Class'])
#abusive_detect_model = MultinomialNB().fit(emails_tfidf,email_data_final['Class'])


# In[242]:


# Checking for 5th email

print('predicted:',abusive_detect_model.predict(tfidf_a_email)[0])
#print('predicted:',abusive_detect_model.predict(X_feature)[0])

print('expected:',email_data_final.Class[0])


# In[243]:


# Model Evaluation

all_predictions = abusive_detect_model.predict(emails_tfidf)
print(all_predictions)


# In[244]:


comapare_df =pd.DataFrame({"predicted":all_predictions,"Actual":email_data_final["Class"] })


# In[245]:


comapare_df.head(20)


# In[246]:


from sklearn.metrics import classification_report,confusion_matrix, accuracy_score


# In[247]:


print(classification_report(email_data_final['Class'],all_predictions))
print(confusion_matrix(email_data_final['Class'],all_predictions))


# In[248]:


print(accuracy_score(email_data_final['Class'],all_predictions))


# ## Data Partitioning

# In[249]:


from sklearn.model_selection import train_test_split


# In[250]:



# Feature data 
X_data = emails_tfidf
#X_data = X_feature


# In[251]:


y =email_data_final["Class"]


# In[252]:


X_data.shape


# In[253]:


y.shape


# In[254]:


from sklearn.model_selection import train_test_split


# In[255]:


X_train, X_test, y_train, y_test = train_test_split(X_data, y, test_size=0.2, random_state=0)


# In[256]:


X_train.shape, y_train.shape, X_test.shape,  y_test.shape


# In[257]:


print(y_train.value_counts())
sns.countplot(y_train)


# #### The Y varable is imbalanced dataset.

# ## Resampling

# #### SMOTE

# In[258]:


from imblearn.over_sampling import SMOTE


# In[259]:


sm=SMOTE(random_state=0)


# In[260]:


X_train_bal, y_train_bal = sm.fit_resample(X_train,y_train)


# In[261]:


X_train_bal.shape, y_train_bal.shape


# In[262]:


print(y_train_bal.value_counts())
sns.countplot(y_train_bal)


# In[263]:


y_train_bal


# #### 2. Underspamling

# In[264]:


from imblearn.under_sampling import RandomUnderSampler


# In[265]:


us = RandomUnderSampler(random_state=42)


# In[266]:


X_bal,y_bal = us.fit_sample(X_feature,y)


# In[ ]:


from collections import Counter


# In[ ]:


print("oringinal dataset shape {}".format(Counter(y)))
print("oringinal dataset shape {}".format(Counter(y_bal)))


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:





# In[ ]:





# In[ ]:





# ##  Model Building

# #### 1. Model Naive Bayes

# In[268]:


from sklearn.naive_bayes import MultinomialNB


# In[269]:



nb_model =MultinomialNB()
nb_model.fit(X_train_bal,y_train_bal)


# In[270]:


from sklearn.metrics import classification_report,confusion_matrix, accuracy_score, f1_score, recall_score, precision_score


# In[271]:


# Train accuracy

y_train_pred = nb_model.predict(X_train_bal)


# In[272]:


train_accuarcy_nb = accuracy_score(y_train_bal,y_train_pred)
train_accuarcy_nb


# In[273]:



# Test Accuracy

y_test_pred = nb_model.predict(X_test)
#print(y_predicted)
#print(np.array(y_test))

test_accuarcy_nb = accuracy_score(y_test,y_test_pred)
test_accuarcy_nb


# In[274]:


train_accuarcy_nb,test_accuarcy_nb


# In[275]:


recall_score_nb = recall_score(y_test,y_test_pred)
recall_score_nb


# In[276]:


precision_score_nb = precision_score(y_test,y_test_pred)
precision_score_nb


# In[277]:


f1_score_nb = f1_score(y_test,y_test_pred)
f1_score_nb


# In[278]:


print(classification_report(y_test,y_test_pred))
print(confusion_matrix(y_test,y_test_pred))


# #### 2.SVM Classifier

# In[279]:


from sklearn.svm import SVC


# In[280]:


sv_model =SVC()
sv_model.fit(X_train_bal,y_train_bal)


# In[281]:


# Train Accuracy
y_train_pred = sv_model.predict(X_train_bal)
train_accur_svm =accuracy_score(y_train_bal,y_train_pred) 


# In[282]:



# Test accuracy
y_test_pred = sv_model.predict(X_test)
test_accu_svm =accuracy_score (y_test,y_test_pred)


# In[283]:



train_accur_svm, test_accu_svm


# In[284]:


recall_score_svm = recall_score(y_test,y_test_pred)
recall_score_svm


# In[285]:


precision_score_svm = precision_score(y_test,y_test_pred)
precision_score_svm


# In[286]:


f1_score_svm = f1_score(y_test,y_test_pred)
f1_score_svm


# In[287]:


print(classification_report(y_test,y_test_pred))
print(confusion_matrix(y_test,y_test_pred))


# #### 3.KNN Classifier

# In[288]:


from sklearn.neighbors import KNeighborsClassifier


# In[289]:


knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_bal,y_train_bal )


# In[290]:


# Train accuaracy
 
y_train_pred = knn_model.predict(X_train_bal)
train_accur_knn =accuracy_score(y_train_bal,y_train_pred)


# In[291]:


# Test accuracy

y_test_pred = knn_model.predict(X_test)
test_accu_knn =accuracy_score (y_test,y_test_pred)


# In[292]:


train_accur_knn, test_accu_knn


# In[293]:


f1_score_knn = f1_score(y_test,y_test_pred)
f1_score_knn


# In[294]:


recall_score_knn = recall_score(y_test,y_test_pred)
recall_score_knn


# In[295]:


precision_score_knn = precision_score(y_test,y_test_pred)
precision_score_knn


# In[296]:


print(classification_report(y_test,y_test_pred))
print(confusion_matrix(y_test,y_test_pred))


# #### 4. Gradient Boosting Classifier

# In[297]:


from sklearn.ensemble import GradientBoostingClassifier


# In[298]:


gb_model = GradientBoostingClassifier()
gb_model.fit(X_train_bal,y_train_bal)


# In[299]:


# Train Accuracy
y_train_pred = gb_model.predict(X_train_bal)
train_accur_gb =accuracy_score(y_train_bal,y_train_pred)


# In[300]:


# Test accuracy
y_test_pred = gb_model.predict(X_test)
test_accu_gb =accuracy_score (y_test,y_test_pred)


# In[301]:


train_accur_gb, test_accu_gb


# In[302]:


recall_score_gb = recall_score(y_test,y_test_pred)
recall_score_gb


# In[303]:


precision_score_gb = precision_score(y_test,y_test_pred)
precision_score_gb


# In[304]:


f1_score_gb = f1_score(y_test,y_test_pred)
f1_score_gb


# In[305]:


print(classification_report(y_test,y_test_pred))
print(confusion_matrix(y_test,y_test_pred))


# #### 6. XGBoost Classifier

# In[306]:


from xgboost import XGBClassifier


# In[307]:


xgb_model = XGBClassifier()
xgb_model.fit(X_train_bal,y_train_bal)


# In[308]:


# Train Accuracy
y_train_pred = xgb_model.predict(X_train_bal)
train_accur_xgb =accuracy_score(y_train_bal,y_train_pred)


# In[309]:


# Test accuracy
y_test_pred = xgb_model.predict(X_test)
test_accu_xgb =accuracy_score (y_test,y_test_pred)


# In[310]:


train_accur_xgb, test_accu_xgb


# In[311]:


recall_score_xgb= recall_score(y_test,y_test_pred)
recall_score_xgb


# In[312]:


precision_score_xgb = precision_score(y_test,y_test_pred)

precision_score_xgb


# In[313]:


f1_score_xgb = f1_score(y_test,y_test_pred)
f1_score_xgb


# In[314]:


print(classification_report(y_test,y_test_pred))
print(confusion_matrix(y_test,y_test_pred))


# ## Model Evaluation

# #### # Model Comparison

# In[315]:




Evaluation_Scores =  {"Models":["Naive Bayes","SVM","KNN","GBM","XGB"],
                    "Train Accuracy":[train_accuarcy_nb, train_accur_svm,train_accur_knn,train_accur_gb, train_accur_xgb],
                    "Test Accuracy":[test_accuarcy_nb,test_accu_svm,test_accu_knn,test_accu_gb,test_accu_xgb],
                    "Recall" :[recall_score_nb,recall_score_svm,recall_score_knn,recall_score_gb,recall_score_xgb],
                    "Precision":[precision_score_nb,precision_score_svm,precision_score_knn,precision_score_gb,precision_score_xgb],
                    "F1 Score":[f1_score_nb,f1_score_svm,f1_score_knn,f1_score_gb,f1_score_xgb]}


# In[316]:


Evaluation_Scores = pd.DataFrame(Evaluation_Scores)
Evaluation_Scores


# #### Predictions

# In[317]:


type(pd.DataFrame(X_train_bal))


# In[318]:


import os
os.chdir("D:\\Project\\EmailClassification")


# In[319]:


import pickle


# In[320]:


pickle.dump(cv, open('cv.pkl', 'wb'))


# In[321]:


pickle.dump(xgb_model, open("abusive.pkl", "wb"))


# In[322]:


"\n Helllo Dear How are You! 2 45@ Runnig".replace('\n'," ")


# In[323]:


def text_cleaning(new_email):
    new_email =new_email.replace('\n'," ")
    
    new_email2= to_lower(new_email)
    new_email3= remove_special_characters(new_email2)
    new_email4= removal_hyperlinks(new_email3)
    new_email5= removal_whitespaces(new_email4)
    new_email6= removal_stopwords(new_email5)
    text = lemmatization(new_email6)
    return text


# In[324]:


# = "\n Helllo Dear How are You! 2 45@ Running "


# In[ ]:





# In[383]:


new_email = input("Please enter a new email:  ")
loaded_model = pickle.load(open("abusive.pkl", "rb"))

def new_email_predict(new_email):
    cleaned_string =text_cleaning(new_email)
    corpus =[cleaned_string]
    new_X_test = cv.transform(corpus)
    print(new_X_test)
    tfidf_transformer=TfidfTransformer().fit(new_X_test)
    emails_tfidf = tfidf_transformer.transform(new_X_test)
    new_y_pred = loaded_model.predict(new_X_test)
    return new_y_pred

a_email= new_email_predict(new_email)[0]
if a_email==0:
  print("Abusive")
else :
  print("Non-Abusive")


# In[326]:


import streamlit as st 

from PIL import Image


# In[384]:


def main():
    st.title("Mail Classification")
    
    html_temp = """
  
    <div style="background-color:Black;padding:10px">
    <h2 style="color:white;text-align:center;">Describe Abusive & Non-Abusive text </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    #Sentence = st.text_input("text","Please Enter a valid sentence")
    Sentence = "dick"
    
    result=""
    a_email= new_email_predict(Sentence)[0]
    print(a_email)
    if a_email==0:
            result = 'Abusive'
    else :
            result = "Non-Abusive"
    print(result)
    if st.button("Predict"):
        a_email= new_email_predict(Sentence)[0]
        if a_email==0:
            result = 'Abusive'
        else :
            result = "Non-Abusive"
            print(result)
    st.success('The output is {}'.format(result))
    if st.button("About"):
        st.text("Lets LEarn")
        st.text("Built with Streamlit")


# In[385]:


if __name__=='__main__':
    main()
    

