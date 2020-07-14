# FAKE-NEWS-DETECTION

## What is fake news?
 Fake news,also known as junk news,pseudo news is a form of news consisting of deliberste disinformation spread via traditional news media or online social media.
 
## About this project:
 This project aims at detecting fake news.
 
 Using sklearn  we build a TfidVectorizer on our dataset.Then we intialize a PassiveAggressiveClassifier and fit the model.At the end the accuracy score and confusion matrix tells
 how our model fares.
 
 ![fake-news-detection](https://user-images.githubusercontent.com/67892708/87383431-d8c91d00-c5b6-11ea-9d07-e47afe96f7f3.jpg)
 
 **What is PassiveAggressiveClassifier?**
 
   PassiveAggressive algorithm are a family of algorithms for large scale learning.It is used to shuffle the training data,when shuffle is set to frame.
   
 **Family of PassiveAggressive are:**
 
 ![classification](https://user-images.githubusercontent.com/67892708/87384293-00b98000-c5b9-11ea-9593-92f67fd2d651.png)
 
 **What is TfidVectorizer?**
 
   The TfidVectorizer will *tokenize* documets ,learn the vocabulary and inverse the document frequency weighting and allow you to encode new documents.
   
   + *Term frequency:* This summarizes how often a given word appears within a document.
   
   + *Inverse Document Frequency:* This downscales words that appear a lot across documents.
   
**What is ConfusionMatrix?**

  A Confusion matrix  is a table that is often used to describe the performance of a classification model(or classifier) on a set of test data for which the true values are known. 


 
