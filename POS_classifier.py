import pandas as pd
import math
import spacy
import string
import nltk
import ssl
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm
s=SentimentIntensityAnalyzer()
nlp=spacy.load("en_core_web_sm")
df=pd.read_csv('IMDB Dataset.csv')
pdf=df.loc[df['sentiment']=='positive'].copy()
ndf=df.loc[df['sentiment']=='negative'].copy()
pdf.reset_index(drop=True)
ndf.reset_index(drop=True)
train=pdf.iloc[0:1250]
rem=ndf.iloc[0:1250]
train=pd.concat([train,rem], ignore_index=True)
test=pdf.iloc[1250:2500]
rem=ndf.iloc[1250:2500]
test=pd.concat([test,rem], ignore_index=True)
neg={}
pos={}
v={}  
ppartsofspeech={}
npartsofspeech={}
for i in range (2500):
    doc=nlp(train.iloc[i]['review'])
    doc=[token for token in doc if token.is_alpha]
    POS=[(token, token.pos_) for token in doc]
    if train.iloc[i]['sentiment']=='negative':    
        for j in POS:
            # if j[1] in ppartsofspeech.keys():
            #     ppartsofspeech[j[1]]+=1
            # else:
            #     ppartsofspeech[j[1]]=1
            if j[1]=='ADJ': #replace with 'ADV' for adverbs and 'VERB' for verbs
                if str(j[0]) in neg.keys():
                    neg[str(j[0])]+=1
                else:
                    neg[str(j[0])]=1
    elif train.iloc[i]['sentiment']=='positive': 
        for j in POS:
            # if j[1] in npartsofspeech.keys():
            #     npartsofspeech[j[1]]+=1
            # else:
            #     npartsofspeech[j[1]]=1
            if j[1]=='ADJ':
                if str(j[0]) in pos.keys():
                    pos[str(j[0])]+=1
                else:
                    pos[str(j[0])]=1
# print(sorted(ppartsofspeech.items(), key=lambda item: item[1], reverse=True))
# print(sorted(npartsofspeech.items(), key=lambda item: item[1], reverse=True))
for word in neg.keys():
    if word in v.keys():
        v[word]+=1
    else:
        v[word]=1
for word in pos.keys():
    if word in v.keys():
        v[word]+=1
    else:
        v[word]=1
V=list(v.keys())
countsneg={}
countspos={}
for word in V:
    if word in neg.keys():
        countsneg[word]=neg[word]
    else:
        countsneg[word]=0
    if word in pos.keys():
        countspos[word]=pos[word]
    else:
        countspos[word]=0
countsumneg=sum(countsneg.values())
negloglikelihood={}
b=countsumneg+len(V)
for key in countsneg.keys():
    a=countsneg[key]+1
    c=a/b
    negloglikelihood[key]=math.log10(c)
countsumpos=sum(countspos.values())
posloglikelihood={}
B=countsumpos+len(V)
for key in countspos.keys():
    A=countspos[key]+1
    C=A/B
    posloglikelihood[key]=math.log10(C)
result=[]
actual=[]  
posses={}
for i in range (2500):
    negsum=math.log10(0.5)
    possum=math.log10(0.5)
    testdoc={}
    doc=nlp(test.iloc[i]['review'])
    doc=[token for token in doc if token.is_alpha]
    POS=[(token, token.pos_) for token in doc]
    if test.iloc[i]['sentiment']=='negative': 
        actual.append("neg")
    elif test.iloc[i]['sentiment']=='positive': 
        actual.append("pos")
    for j in POS:
        if j[1] in posses.keys():
            posses[j[1]]+=1
        else:
            posses[j[1]]=1
        if str(j[0])in testdoc.keys() and j[1]=='ADJ':
            testdoc[str(j[0])]+=1
        else:
            testdoc[str(j[0])]=1 
    for word in testdoc.keys():
        if word in V:
            negsum+=negloglikelihood[word]
            possum+=posloglikelihood[word]     
    if possum>negsum:
        result.append("pos") 
    else:
        result.append("neg")
# print(sorted(posses.items(), key=lambda item: item[1], reverse=True))
# exit(0)
tp=0
tn=0
fp=0
fn=0
for i in range(len(result)):
    if result[i]==actual[i]:
        if result[i]=="pos":
            tp+=1
        elif result[i]=="neg":
            tn+=1
    else:
        if result[i]=="pos" and actual[i]=="neg":
            fp+=1
        elif result[i]=="neg" and actual[i]=="pos":
            fn+=1
precision=tp/(tp+fp)
recall=tp/(tp+fn)
accuracy=(tp+tn)/(tp+fp+tn+fn)
f1score=(2*precision*recall)/(precision+recall)
print(tp,fp,fn,tn)
print(precision,recall,accuracy,f1score)

#running the same classifier on the assignment-5 dataset

# import os
# import math
# import numpy as np
# import spacy
# nlp=spacy.load("en_core_web_sm")
# path="/mnt/c/Users/Nikhil/Desktop/train/train"
# neg={}
# pos={}
# v={}
# space=" "
# for filename in os.listdir(path):
#     if os.path.isfile(os.path.join(path, filename)):
#         with open(os.path.join(path, filename), 'r') as f:
#             lines=f.read()
#             doc=nlp(lines)
#             doc=[token for token in doc if token.is_alpha]
#             POS=[(token, token.pos_) for token in doc]   
#             if filename[:3]=="neg":    
#                 for i in POS:
#                     if i[1]=='ADV':
#                         if str(i[0]) in neg.keys():
#                             neg[str(i[0])]+=1
#                         else:
#                             neg[str(i[0])]=1
#             elif filename[:3]=="pos":
#                 for i in POS:
#                     if i[1]=='ADV':
#                         if str(i[0]) in pos.keys():
#                             pos[str(i[0])]+=1
#                         else:
#                             pos[str(i[0])]=1
# for word in neg.keys():
#     if word in v.keys():
#         v[word]+=1
#     else:
#         v[word]=1
# for word in pos.keys():
#     if word in v.keys():
#         v[word]+=1
#     else:
#         v[word]=1
# V=list(v.keys())
# countsneg={}
# countspos={}
# for word in V:
#     if word in neg.keys():
#         countsneg[word]=neg[word]
#     else:
#         countsneg[word]=0
#     if word in pos.keys():
#         countspos[word]=pos[word]
#     else:
#         countspos[word]=0
# countsumneg=sum(countsneg.values())
# negloglikelihood={}
# b=countsumneg+len(V)
# for key in countsneg.keys():
#     a=countsneg[key]+1
#     c=a/b
#     negloglikelihood[key]=math.log10(c)
# countsumpos=sum(countspos.values())
# posloglikelihood={}
# B=countsumpos+len(V)
# for key in countspos.keys():
#     A=countspos[key]+1
#     C=A/B
#     posloglikelihood[key]=math.log10(C)
# newpath="/mnt/c/Users/Nikhil/Desktop/test/test"
# result=[]
# actual=[]

# for filename in os.listdir(newpath):
#     negsum=math.log10(0.5)
#     possum=math.log10(0.5)
#     testdoc=[]
#     if os.path.isfile(os.path.join(newpath, filename)):
#         if filename[:3]=="neg":
#             actual.append("neg")
#         else:
#             actual.append("pos")
#         with open(os.path.join(newpath, filename), 'r') as f:
#             lines=f.read()
#             doc=nlp(lines)
#             doc=[token for token in doc if token.is_alpha]
#             POS=[(token, token.pos_) for token in doc]
#             for i in POS:
#                 if i[1]=='ADV':
#                     testdoc.append(str(i[0]))
#     unique_testdoc={}
#     for word in testdoc:
#         if word in unique_testdoc.keys():
#             unique_testdoc[word]+=1
#         else:
#             unique_testdoc[word]=1
#     for word in unique_testdoc.keys():
#         if word in V:
#             negsum+=negloglikelihood[word]
#             possum+=posloglikelihood[word]     
#     if possum>negsum:
#         result.append("pos") 
#     else:
#         result.append("neg")

# tp=0
# tn=0
# fp=0
# fn=0
# for i in range(len(result)):
#     if result[i]==actual[i]:
#         if result[i]=="pos":
#             tp+=1
#         elif result[i]=="neg":
#             tn+=1
#     else:
#         if result[i]=="pos" and actual[i]=="neg":
#             fp+=1
#         elif result[i]=="neg" and actual[i]=="pos":
#             fn+=1
# precision=tp/(tp+fp)
# recall=tp/(tp+fn)
# accuracy=(tp+tn)/(tp+fp+tn+fn)
# f1score=(2*precision*recall)/(precision+recall)
# print(tp,fp,fn,tn)
# print(precision,recall,accuracy,f1score)