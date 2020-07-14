# FAKE-NEWS-DETECTION

## What is fake news?
 Fake news,also known as junk news,pseudo news is a form of news consisting of deliberste disinformation spread via traditional news media or online social media.
 
## About this project:
 This project aims at detecting fake news.
 
 Using sklearn  we build a TfidVectorizer on our dataset.Then we intialize a PassiveAggressiveClassifier and fit the model.At the end the accuracy score and confusion matrix tells
 how our model fares.
 
 ![fake-news-detection](https://user-images.githubusercontent.com/67892708/87383431-d8c91d00-c5b6-11ea-9d07-e47afe96f7f3.jpg)
 
#### **What is PassiveAggressiveClassifier?**
 
  PassiveAggressive algorithm are a family of algorithms for large scale learning.It is used to shuffle the training data,when shuffle is set to frame.
   
#### **Family of PassiveAggressive are:**
 
 ![classification](https://user-images.githubusercontent.com/67892708/87384293-00b98000-c5b9-11ea-9593-92f67fd2d651.png)
 
#### **What is TfidVectorizer?**
 
  The TfidVectorizer will *tokenize* documets ,learn the vocabulary and inverse the document frequency weighting and allow you to encode new documents.
   
   + *Term frequency:* This summarizes how often a given word appears within a document.
   
   + *Inverse Document Frequency:* This downscales words that appear a lot across documents.
   
#### **What is ConfusionMatrix?**
   A Confusion matrix  is a table that is often used to describe the performance of a classification model(or classifier) on a set of test data for which the true values are known. 
  
![Confusion_Matrix1_1](https://user-images.githubusercontent.com/67892708/87385285-555dfa80-c5bb-11ea-9674-6dc3a1b7da46.png)

**Definition of these terms:**

  + *Positive(P):* Observation is positive.
  
  + *Negative(N):* Observation is negative.
  
  + *True Positive(TP):* Observation is positive,and is predicted to be positive.
  
  + *False Negative(FN):* Observation is positive,but is predicted negative.
  
  + *True Negative(TN):* Observation is negative,and is predicted to be positive.
  
  + *False Positive(FP):* Observation is negative,but is predicted positive.
  
#### **Accuracy is computed as belows:**

![accuracy](https://user-images.githubusercontent.com/67892708/87388253-11222880-c5c2-11ea-8d6c-42e14bd8dd0b.png)

*Now let's get into coding!!!*

**Install all the necessary libraries using pip.**

```
!pip install numpy pandas sklearn
```
**Make all the necessary imports**

```
import numpy as np
import pandas as pd
import intertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
```

**Now read the data**

```
dataframe=pd.read_csv("news.csv)
dataframe
```

**What is .info() function?**

The info function is used to print a concise summary of a dataframe.This method prints information about a dataFrame.This method prints information about a DataFrame including the index dtype and column ,non-null values and memory usage.

```
dataframe.info()
```








 
