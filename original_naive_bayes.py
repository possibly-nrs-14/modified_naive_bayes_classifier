import pandas as pd
import math
import spacy
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
for i in range (2500):
    doc=nlp(train.iloc[i]['review'])
    doc=[token for token in doc if token.is_alpha]
    if train.iloc[i]['sentiment']=='negative':    
        for word in doc:
            if str(word)in neg.keys():
                neg[str(word)]+=1
            else:
                neg[str(word)]=1
    elif train.iloc[i]['sentiment']=='positive': 
        for word in doc:
            if str(word)in pos.keys():
                pos[str(word)]+=1
            else:
                pos[str(word)]=1   
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
# sentence="I liked most of the dialogue, I liked the cast, I thought it was well acted. I particularly enjoyed Ellen DeGeneres' perfect deadpan performance.<br /><br />What didn't work for me was: (1) the drawn-out affair with the younger man (too long, too seemingly out of character for Helen), (2) the seemingly endless cinematic cliches, mostly visual but including interminable voiced over re-readings of the love letter itself (its contents should have a mystery); (3) a young woman feminist-scholar and, ironically, a fireworks scene (no wonder this reminded me of that horrid How to Make an American Quilt movie); (4) the bumbling gotcha cop who smells dope everywhere (no cliche there either!); and (5) a nauseatingly romanticized small town setting.<br /><br />I would have preferred the film to more persuasively explore the source of (or even glorify) Helen's bitterness, to have included much more of DeGeneres' character, to have eliminated or reduced the various intergenerational artifices, and to be a little less uncritical of small town life.<br /><br />Had it been developed as a play first, those criticisms might have been addressed before committing the material to this film, which unfortunately is decidedly mediocre"
# doc=nlp(sentence)
# doc=[token for token in doc if token.is_alpha]
# testdoc={}
# negsum=math.log10(0.5)
# possum=math.log10(0.5)
# for word in doc:
#     if str(word)in testdoc.keys():
#         testdoc[str(word)]+=1
#     else:
#         testdoc[str(word)]=1 
# for word in testdoc.keys():
#     if word in V:
#         print(word,posloglikelihood[word],negloglikelihood[word])
#         negsum+=negloglikelihood[word]
#         possum+=posloglikelihood[word]
# print(possum,negsum)

result=[]
actual=[]  
for i in range (2500):
    negsum=math.log10(0.5)
    possum=math.log10(0.5)
    testdoc={}
    doc=nlp(test.iloc[i]['review'])
    doc=[token for token in doc if token.is_alpha]
    if test.iloc[i]['sentiment']=='negative': 
        actual.append("neg")
    elif test.iloc[i]['sentiment']=='positive': 
        actual.append("pos")
    for word in doc:
        if str(word)in testdoc.keys():
            testdoc[str(word)]+=1
        else:
            testdoc[str(word)]=1 
    for word in testdoc.keys():
        if word in V:
            negsum+=negloglikelihood[word]
            possum+=posloglikelihood[word]     
    if possum>negsum:
        result.append("pos") 
    else:
        result.append("neg")
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




