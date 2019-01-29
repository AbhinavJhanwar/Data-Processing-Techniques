'''
Created on Jun 11, 2018

@author: abhinav.jhanwar
'''

import re
from nltk.tag import pos_tag
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer, WordNetLemmatizer
import string
# Laoding libraries -Utility
from functools import lru_cache

# =============================================================================
# Text Processing
# removing punctuation
# lemmatizing & stemming data
# =============================================================================

#Defining lematizer, pos tagger and stemmer
wnl     = WordNetLemmatizer()
porter  = PorterStemmer()
##############################
#memonization - to increase speed n efficiency
lemmatize_mem = lru_cache(maxsize=16384)(wnl.lemmatize)
###############################
def tree_2_word(treebank_tag):
    if treebank_tag.startswith('J'):
        return wn.ADJ
    elif treebank_tag.startswith('V'):
        return wn.VERB
    elif treebank_tag.startswith('R'):
        return wn.ADV
    else:
        return wn.NOUN
    
def lemmatize(word,pos=wn.NOUN):
    return(lemmatize_mem(word,pos))
     
def text_process(t,pos=True,lemmetize=True,stem=True):
    # Defining stop words
    stops = set(stopwords.words("english"))
    # Remove linebreak, tab, return               
    t = re.sub('[\n\t\r]+',' ',t)          
    # Convert to lower case
    t = t.lower()             
    # Remove Non-letters                            
    t = re.sub('[^a-zA-Z]',' ',t)  
    # Sentence wise tokenize: required otherwise pos tagging will not work properly                       
    sentence = nltk.sent_tokenize(t)                      
    modified_sentence=""
    for s in sentence:
        # Word Tokenization 
        words = nltk.word_tokenize(s)     
        # checking if pos is required                
        if pos:               
            # Part of speech Tagging                            
            tag_words = pos_tag(words)                        
            lemmatized= " "
            for tw in tag_words:
                # checking stop word
                if tw[0] not in stops:
                    # lemmatization
                    if lemmetize:                         
                        lemma=lemmatize(tw[0],tree_2_word(tw[1]))  
                        # check if lemmetizer word is present in the wordnet library if not present then stemming is required
                        # example running not in synsets: stem to run  
                        # stemming check if word is in library then perform otherwise skip
                        if (stem) & (wn.synsets(lemma)==[]) & (wn.synsets(porter.stem(lemma))!=[]):
                            # Stemming
                            lemma = porter.stem(tw[0])    
                    else:
                        if stem:
                            lemma=porter.stem(tw[0])
                            if not wn.synsets(lemma):
                                lemma=tw[0]
                        else:
                            lemma=tw[0]
                    lemmatized=lemmatized+" "+lemma
        else:
            lemmatized= " "
            for w in words:
                if w not in stops:
                    if lemmetize:                         # lemmatization
                        lemma=lemmatize(w)
                        if (stem) & (wn.synsets(lemma)==[]) & (wn.synsets(porter.stem(lemma))!=[]):
                            # Stemming
                            lemma = porter.stem(w)    
                    else:
                        if stem:
                            lemma=porter.stem(w)
                            if not wn.synsets(lemma):
                                lemma=w
                        else:
                            lemma=w
                    lemmatized=lemmatized+" "+lemma               
        modified_sentence=modified_sentence+" "+lemmatized
    # Remove Punctuations  
    t = re.sub('['+string.punctuation+']+','',\
               modified_sentence)                            
    t = re.sub('\s+\s+',' ',t)                             # Remove double whitespace
    t= re.sub(r'\\', ' ', t)                                # Remove \ slash
    t= re.sub(r'\/', ' ', t)                                # Remove / slash
    t = t[1:]
    #print(t)
    return(t)
