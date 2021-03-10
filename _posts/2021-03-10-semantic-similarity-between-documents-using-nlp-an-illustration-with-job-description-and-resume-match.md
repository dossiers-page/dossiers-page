---
title: Semantic Similarity  between documents using NLP, an illustration with job
  description and resume match
layout: single
classes: wide
---

Suppose that there are many Job Descriptions and you want to know the closest match of your profile against them. The normal way is to search for keywords and manually go through the words.  An alternate approach uses Natural Language Processing models that can tell the distance between words (or documents). 

One of the best approaches to find semantic similarity between words started with the work  "[A Neural Probabilistic Language Model](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)" by Yoshua Bengio ( 2003). The idea was straightforward. Make a neural network that takes a bag of k words and predicts the next word. After going through large corpora, the weights of the model will represent the words based on their semantic similarity.  Thus the "representation" of knowledge by the neural network is the best representation of the word.  The parameters in such a representation would be the number of simultaneous words in the bag and the vector's size (8 vectors/ 16 vectors/ 32 vectors etc.).   

This in layman's sense, is like using a PINCODE/Latitude-Longitude. Closer values would be closer to each other.  This also means that words like bank (**homonyms**) will have almost equal probability to see river and cash as next words. 

**"Efficient Estimation of Word Representations in Vector Space"** was the next breakthrough in this aspect, and word2vec as a library became the most popular NLP library. 

This demonstration uses word2vec trained on job description datasets to show how efficiently it can capture the semantic similarity between keywords used and show the distance between them (lower the distance, better match).


### Please wait for the below application to load 
<iframe src="https://pythonapps.dossiers.page:8443/" title="Resume-JD Matching" width='100%' height='1000' frameBorder="0"  allowtransparency="true"></iframe>
