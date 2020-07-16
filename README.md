# FAKE-NEWS-DETECTION

## What is fake news?
 Fake news, also known as junk news, pseudo news is a form of news consisting of deliberste disinformation spread via traditional news media or online social media.
 
## About this project:
 This project aims at detecting fake news.
 
 Using sklearn  we build a *TfidVectorizer* on our dataset.Then we intialize a *PassiveAggressiveClassifier* and fit the model. At the end the *accuracy score* and *confusion matrix* tells how our model fares.
 
 ![fake-news-detection](https://user-images.githubusercontent.com/67892708/87383431-d8c91d00-c5b6-11ea-9d07-e47afe96f7f3.jpg)
 
## The DataSet:
 
 The dataset that we use here is **news.csv**.This dataset takes upto 29.2MB of space and you can
  [download it here.](https://drive.google.com/file/d/1er9NJTLUA3qnRuyhfzuN0XUsoIC4a-_q/view)
 
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
  
  + *True Negative(TN):* Observation is negative,and is predicted to be negative.
  
  + *False Positive(FP):* Observation is negative,but is predicted positive.
  
#### **Accuracy is computed as shown below:**

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

**What is .keys() function?**

The keys function returns the *'info axis'* for the pandas object.If the pandas object is series then it returns index.If the pandas object is dataFrame then it retrurns columns.

```
dataframe.keys()
```

**To get the shape of the object**

```
dataframe.shape
```

**What is .describe() function?**

The describe method computes a summary of statistics pertaining to the DataFrame columns.

```
dataframe.describe()
```

**What is .head() function?**

This function returns the first n rows for the object based on position.It is useful for quickly testing if your object has right type of data in it.

```
dataframe.head()
```

**Labelling**

Data labelling is important because the machine learning algorithm has to understand the data.

```
labels=dataframe.label
labels.head()
```

**Split the dataset into Training and Testing**

 + *Training set:* A subset to train a model.
 
 + *Test set:* A subset to test the trained model.

![Screenshot (182)](https://user-images.githubusercontent.com/67892708/87423405-e18e1300-c5f7-11ea-80c5-defcb1aa97ae.png)

**Split the dataset**

```
x_train,x_test,,y_train,y_test=train_test_split(dataframe['text'],labels,test_size=0.2,random_state=7)
```

**TfidVectorizer**

*Initialize a TfidVectorizer.*

```
tfidf_vectorizer=TfidVectorizer(stop_words='english',max_df=0.7)
```

*Fit and transform train set,transform test set.*

```
tfidf_train=tfidf_vectorizer.fit_transform(x_train)
tfidf_test=tfidf_vectorizer.transform(x_test)
```

**PassiveAggressiveClassifier**

*Initialize a PassiveAggressiveClassifier.*

```
pac=PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)
```

*Predict on the test set and calculate the accuracy.*

```
y_pred=pac.predict(tfidf_test)
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {rund(score*100,2)}%')
```

The accuracy obtained with this model is 93.05%.

**Printing the confusion matrix**

```
confusion_matrix(y_test,y_pred,labels=['FAKE','REAL'])
```

```
OUTPUT:
array([[592, 46],
       [ 42, 587]],dtype=int64)
       
```

Therefore from this model, according to confusion matrix ,we have 592 true positives,587 true negatives, 42 false positives and 46 false negatives.

#### **CONCLUSION**

Using this model, to detect fake news, we have obtained an accuracy of 93.05%.



**Reference:** *DLITHE*
**Website:** www.dlithe.com
*Assignment during Online Internship with Dlithe*[www.dlithe.com](www.dlithe.com)








 
