
# coding: utf-8

# In[1]:


# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 20:13:10 2018

@author: qianhuil
"""
import pymongo
from pymongo import MongoClient
import json
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.collocations import *
import string
import pickle


# In[2]:






def read_in_fromdb():
    #connect to mongodb, start client session
    client = MongoClient()
    #connect to specific database
    db = client.olive
    #connect to specific collection
    collection = db.yelp_reviews
    #finding needs from full collection, data = collection.find({'restaurant_id':'neptune-oyster-boston'}) will find only neptune, etc.
    data = collection.find() #finds all

    #extract data from cursor
    itembuffer=[]
    for item in data:
         itembuffer.append(item)
    return itembuffer


def tokenize(set_data):
    '''
    read the data, tokenize, clean the word
    return a list of strings of reviewing words.
    '''
#     with open(filename, 'r') as f:
#         set_data = json.load(f)


    rate_dict = {}
    for i in set_data:
        rate_dict.setdefault(i["review_rating"],[]).append(i["review_detail"])
    review = [] # list to store rating views
    for i in rate_dict.keys():
        for value in rate_dict[i]:
            for token in value.split():
                if token.isalpha():
                    token = token.lower().strip()
                    token = token.replace("!","")
                    token = token.replace(".","")
                    token = token.replace("?",'')
                    token = token.replace('"','')
                    review.append(token)
        print("appending tokens for rating = {} successful".format(i))
    return review


def freq(set_data):
    '''
    get the frequency dictionary for the word of bag model,
    set differences done, token position changed, not suitable for the N-gram
    '''

    freq_disk = nltk.FreqDist(tokenize(set_data))
           # review = [t.lower() for t in value.split()]
#            for j in review:
#                review.append(j.lower())
#        print(review)
#            review.append(word.lower() for word in word_tokenize(value))
        #print(review)

        #clean_review = review[:]
#    print(review[:100])
    clean_freq_disk = freq_disk.copy()

    reviewset = set(freq_disk.keys())
    stopword = set(stopwords.words("english"))
    remove_word = reviewset & stopword

    for token in freq_disk.keys():
        if token in remove_word:
            del clean_freq_disk[token]
    print("stopword cleaning success")

    return clean_freq_disk

def clean_token(set_data):
    '''
    clean stopwords for N-gram
    '''
    tokens = tokenize(set_data)
    stop_words = set(stopwords.words("english"))
    clean_token = [w for w in tokens if not w in stop_words]
    return clean_token

def feature(set_data,feature_size = 2000):

#        for key,val in freq.items():
#            print(str(key)+":"+str(val))

    word_features = list(freq(set_data))[:feature_size]
    return word_features    # a list of top 100 frequent word

def feature_2(set_data,feature_size = 100):
    '''
    getting the feature using bigram
    '''
    stop_words = set(stopwords.words("english"))
    tokens = tokenize(set_data)
    bigram_measures = nltk.collocations.BigramAssocMeasures()    # bigram
#    trigram_measures = nltk.collocations.TrigramAssocMeasures()     # trigram
    finder = BigramCollocationFinder.from_words(tokens)
    finder.apply_word_filter(lambda w:w in stop_words)##what kind of filter suits here>???

    finder.apply_freq_filter(2)
    scored = finder.score_ngrams(bigram_measures.raw_freq)
    word_features_bigram = finder.nbest(bigram_measures.raw_freq, feature_size)
    bigramFeature = []
    for item in word_features_bigram:
        bigramFeature.append(" ".join(item))

    print(bigramFeature[:100])
    return bigramFeature

def feature_3(set_data,feature_size = 100):
    '''
    getting the feature using trigram
    '''
    stop_words = set(stopwords.words("english"))
    tokens = tokenize(set_data)        ##!!trigram keep the stopwords,  A and B


    trigram_measures = nltk.collocations.TrigramAssocMeasures()     # trigram
    finder = TrigramCollocationFinder.from_words(tokens)
    finder.apply_word_filter(lambda w:w in stop_words)
    finder.apply_ngram_filter(lambda w1, w2, w3: 'and' in (w1, w3))##what kind of filter suits here>???
    finder.apply_freq_filter(2)
    scored = finder.score_ngrams(trigram_measures.raw_freq)
    word_features_trigram = finder.nbest(trigram_measures.raw_freq, feature_size)
    trigramFeature=[]
    for item in word_features_trigram:
        trigramFeature.append(' '.join(item))
    print(trigramFeature[:100])
    return trigramFeature

def document_features(word_features,review):
    features = {}
    for word in word_features:
        features["contains({})".format(word)] = review.count(word)
    return features


def document_features2(word_features,review):
    features = {}
    for word in word_features:
        if word in review:
            features["contains({})".format(word)] = "True"
        else:
            features["contains({})".format(word)] = "False"
    return features



def train_model_count(feature_function,feature_size):
    '''
    for N-gram
    '''

    set_data = read_in_fromdb()

#     with open(filename, 'r') as f:
#         set_data = json.load(f)

    word_features = feature_function(set_data,feature_size)

    featuresets = []
    for i in set_data:

#        print(type(word_tokenize(i["review_detail"])))
#         token = []
# #        print(type(token))
#         for word in word_tokenize(i["review_detail"]):
#             token.append(word.lower())
        #token.append(word.lower() for word in word_tokenize(i["review_detail"]))
#         label = (document_features(word_features, token),i["review_rating"])
        label = (document_features(word_features, i["review_detail"]),i["review_rating"])
        featuresets.append(label)
    train_set,test_set = featuresets[100:],featuresets[:100]
    classifier = nltk.NaiveBayesClassifier.train(train_set)


    print(nltk.classify.accuracy(classifier, test_set))
    classifier.show_most_informative_features(30)


def train_model_count_words(feature_function,feature_size):

    set_data = read_in_fromdb()

#     with open(filename, 'r') as f:
#         set_data = json.load(f)

    word_features = feature_function(set_data,feature_size)

    featuresets = []
    for i in set_data:

#        print(type(word_tokenize(i["review_detail"])))
        token = []
# #        print(type(token))
        for word in word_tokenize(i["review_detail"]):
            token.append(word.lower())
        #token.append(word.lower() for word in word_tokenize(i["review_detail"]))
        label = (document_features(word_features, token),i["review_rating"])
#         label = (document_features(word_features, i["review_detail"]),i["review_rating"])
        featuresets.append(label)
    train_set,test_set = featuresets[100:],featuresets[:100]
    classifier = nltk.NaiveBayesClassifier.train(train_set)


    print(nltk.classify.accuracy(classifier, test_set))
    classifier.show_most_informative_features(30)



def train_model_yesno(feature_function,feature_size):

    set_data = read_in_fromdb()

#     with open(filename, 'r') as f:
#         set_data = json.load(f)

    word_features = feature_function(set_data,feature_size)

    featuresets = []
    for i in set_data:

#        print(type(word_tokenize(i["review_detail"])))
        token = []
# #        print(type(token))
#         for word in word_tokenize(i["review_detail"]):
#             token.append(word.lower())
        #token.append(word.lower() for word in word_tokenize(i["review_detail"]))
#         label = (document_features(word_features, token),i["review_rating"])
#         label = (document_features(word_features, i["review_detail"]),i["review_rating"])
        label = (document_features2(word_features, i["review_detail"]),i["review_rating"])
        featuresets.append(label)
    train_set,test_set = featuresets[100:],featuresets[:100]
    classifier = nltk.NaiveBayesClassifier.train(train_set)


    print(nltk.classify.accuracy(classifier, test_set))
    classifier.show_most_informative_features(30)

    return classifier



def train_classifier(feature_size1, feature_size2=100):
    #read in data to set_data

    set_data = read_in_fromdb()

    #extract features from feature function passed in
    print("extracting the features for corpus")
    word_features = feature(set_data, feature_size1) #list of features, either words or n-grams
    word_features2 = feature_2(set_data, feature_size2)
    word_features3 = feature_3(set_data, feature_size2)


    featuresets = []
    for i in set_data:
        label = (document_features(word_features, i["review_detail"]),i["review_rating"])
        label2 = (document_features2(word_features2, i["review_detail"]),i["review_rating"])
        label3 = (document_features3(word_features3, i["review_detail"]),i["review_rating"])
        featuresets.append(label)
        featuresets.append(label2)
        featuresets.append(label3)
    train_set,test_set = featuresets[100:],featuresets[:100]
    classifier = nltk.NaiveBayesClassifier.train(train_set)


    print(nltk.classify.accuracy(classifier, test_set))
    classifier.show_most_informative_features(30)

    return classifier






# In[11]:


# In[ ]:

if __name__ == "__main__":
    feature_sizes=[2000, 3000, 4000, 5000]
    for feature_size in feature_sizes:
        classifier = train_model_yesno(feature, feature_size)

        outfile = open('classifier_np_ft{}_bin.pickle'.format(feature_size),"wb")
        pickle.dump(classifier,outfile)
        outfile.close()
        print("feature_size",feature_size)


# In[10]:


#classifier.most_informative_features(n=10)


# In[52]:


#classifier.show_most_informative_features(n=10)


# In[46]:


#train_model_count_words(feature, 3000, "neptune_oyster_test_data.json")


# In[33]:


#train_model_count(feature_2, 500, "neptune_oyster_test_data.json")


# In[26]:


#train_model_count(feature, 10, "full_reviews.json")
