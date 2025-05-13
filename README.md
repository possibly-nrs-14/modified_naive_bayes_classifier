# Sentiment Analysis of Movie Reviews - Naive Bayes Project

This repository contains the implementation of sentiment analysis on movie reviews using a Naive Bayes classifier with various features, as detailed in the included project report.

## Project Report (Verbatim)

Sentiment analysis of movie reviews using a naive

bayes classifier with various features

Nikhil Raj Sriram (Roll no. 2022114003)

December 2023

1 Introduction

Sentiment analysis refers to the task of analyzing the sentiment expressed by
particularly text. It finds great use in various mainstream statistical analyses.
For example, companies often depend on sentiment analysis to gauge whether
their products are doing well in the market or not. In this project, we will be
conducting sentiment analysis on movie reviews.
A popular approach used for sentiment analysis is the ”naive Bayes” approach.
Given a set of documents to be classified, this involves representing each docu-
ment as a bag of words, calculating the probabilities of occurrence of the words
(for each class) independently and ”naively” multiplying them to get the final
likelihood for each class, after which the class with the higher likelihood is con-
sidered. Formally,

P (Ck|x) = P (Ck)
∏n

i=1 P (wi|Ck)

or, equivalently, taking log for faster computation and using add-1 smoothing,

logP (Ck|x) = logP (Ck) +
∑n

i=1 log
(

countwi|Ck
+1

countCk
+|V |

)
The feature used to calculate the probability of each word is typically the fre-
quency of that word;however, in this project, we will be exploring other features
to see how they affect the decision-making of the naive bayes classifier. Specifi-
cally, we will be looking at the following 2 types of features, and analyzing the
results obtained:

1. Linguistically motivated feature: POS(parts-of-speech) tags
2. Features derived from a polarity lexicon VADER

1.1 Assumptions and Dataset used

All naive bayes classifiers implemented in this project use the feature of ”bi-
narization”, i.e, only unique words in each test document are considered for

1

overall word counts during testing. This is because we observed that naive
bayes classifiers with binarization consistently gave higher accuracies and bet-
ter performance than those without binarization.
The primary dataset used for this project is the IMDB movie reviews dataset
from the website kaggle.com. The dataset is not biased (i.e., it has an equal
number of positive and negative reviews). Although this dataset contains 50000
movie reviews, for our purposes, we have used only 5000 reviews of these for
training and testing, keeping in mind the amount of time it takes to run our
classifiers on a dataset of this size. Additionally, we have used half of the chosen
reviews for training and the other half for testing.

2 Naive bayes approach with POS(parts-of-speech)
tags

Y. Wang(2017) proposed the idea of optimizing the traditional näıve bayes clas-
sifier by only considering frequencies of words of certain parts-of-speech for both
training and test data in order to save computational resources that were being
wasted by the traditional approach. He considered using only verbs and adjec-
tives, only adjectives and adverbs, and all three. For this project, we will look
at use of each of verbs, adjectives, and adverbs individually; the aim is to figure
out which part-of-speech works best as a feature for the näıve bayes classifier.

2.1 Observations

Actual Positive    Actual Negative
Predicted Positive 915               184
Predicted Negative 335              1066

Table 1: Confusion matrix of classifier using only adjectives

Metric             Value
Precision          0.83257506
Recall             0.732
Accuracy           0.7924
F1-Score           0.779054916

Table 2: Summary of Classification Metrics

Actual Positive    Actual Negative
Predicted Positive 957               595
Predicted Negative 293               655

Table 3: Confusion matrix of classifier using only adverbs

Metric             Value
Precision          0.616623711
Recall             0.7656
Accuracy           0.6448
F1-Score           0.68308351

Table 4: Summary of Classification Metrics

Actual Positive    Actual Negative
Predicted Positive 779               297
Predicted Negative 471               953

Table 5: Confusion matrix of classifier using only verbs

Metric             Value
Precision          0.72397769
Recall             0.6232
Accuracy           0.6928
F1-Score           0.6698194

Table 6: Summary of Classification Metrics

3 Naive bayes approach with additional features
from polarity lexicon VADER

VADER (Valence Aware Dictionary and sEntiment Reasoner) is a lexicon and
rule-based sentiment analysis tool that is specifically attuned to sentiments ex-
pressed in social media. To each word it assigns a dictionary consisting of in-
formation about the word: whether the word has a positive, negative or neutral
sentiment, and the compound score of the word. The positive, negative and neu-
tral keys of the dictionary are marked 1.0 if the word is positive/negative/neutral
and 0.0 otherwise. The compound score is between -1 (most extreme negative)
and +1 (most extreme positive).
As per the documentation of VADER, positive sentiment corresponds to a com-
pound score >= 0.5, neutral sentiment corresponds to a compound score > −0.5
and < 0.5, and negative sentiment corresponds to a compound score <= −0.5.
However, due to our training dataset having reviews labelled as only “positive”
and “negative”, in our use of VADER for sentiment analysis, we have assumed
that positive sentiment corresponds to a compound score of above 0 and neg-
ative sentiment corresponds to a compound score of less than 0. Additionally,
we found that omitting the “neutral” category alone, and assuming positive
sentiment: compound score >= 0.5 and negative sentiment: compound score
<= −0.5 made little to no difference in the output.

3.1 Incorporating features from VADER

The main aim is the following: To combine the existing features of the naive
bayes classifier and the features in VADER in such a way that they complement
eachother and increase the accuracy of our classifier. We achieved this by scaling
the word frequencies of each word by parameters derived from VADER before
computing the log likelihoods. We did this in 2 ways:

3.1.1 Scaling according to polarity score

Here, we scaled the word frequencies according to the polarity score assigned
by VADER, i.e, the ”compound” score between -1 and +1.
In our naive bayes classifier, each word in the vocabulary has a positive fre-
quency (no. of occurrences of the word in positive training data) and a negative
frequency (no. of occurrences of the word in negative training data). For each
word, we first look at its negative frequency. If the word has a non-zero polarity
score (i.e, if it is present in VADER), we check if the compound score is positive
or negative. Let the compound score (polarity score) be denoted by x. If the
polarity score is negative, we scale up the negative frequency of the word by a
factor of x, i.e., we multiply the negative frequency by 1+|x|. If the polarity
score is positive, we scale down the negative frequency of the word by a factor
of x, i.e., we multiply the negative frequency by 1-|x|. Note that we used |x|
instead of x in this process because we are only interested in the magnitude of
positivity or negativity.

A similar procedure is carried out for the positive frequency of the word; if the
word has a non-zero polarity score , we check if the polarity score is positive or
negative. If the polarity score is positive, we scale up the positive frequency of
the word by a factor of x, i.e., we multiply the positive frequency by 1+|x|. If
the polarity score is negative, we scale down the positive frequency of the word---
