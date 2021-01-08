# Sentiment_Analysis

## Overview

- Analyzed Amazon video game reviews by classifying them as positive or negative
- Tokenized the reviews using 2 different vectorizers, CountVectorizer and TfidfVectorizer
- Applied naive bayes, logistic regression and support vector machine (SVM) on the 2 different vectorizers
- Computed their accuracies, precision, recall and f1-scores
- Naive Bayes with CountVectorizer had highest f1-score of 88.5%

## Tools Used

- Language: Python 
- Packages: pandas, numpy, re, string, matplotlib, seaborn, wordcloud, seaborn, sklearn, nltk
- Data: [Amazon Reviews](https://s3.amazonaws.com/amazon-reviews-pds/tsv/index.txt), download link [here](https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Video_Games_v1_00.tsv.gz)
- Topics: Python, Sentiment Analysis, NLP, TFIDF, Naive Bayes, Logistic Regression, SVM

## Data

The data contains a collection of video game reviews written in the Amazon.com marketplace and associated metadata from 1995 until 2015 and can be downloaded [here](https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Video_Games_v1_00.tsv.gz). It contains the following columns:

| Variable          | Description                                                                                                    |
|-------------------|:---------------------------------------------------------------------------------------------------------------|
| marketplace       | 2 letter country code of the marketplace where the review was written.                                         |
| customer_id       | Random identifier that can be used to aggregate reviews written by a single author.                            |
| review_id         | The unique ID of the review.                                                                                   |
| product_id        | The unique Product ID the review pertains to.                                                                  |
| product_parent    | Random identifier that can be used to aggregate reviews for the same product.                                  |
| product_title     | Title of the product.                                                                                          |
| product_category  | Broad product category that can be used to group reviews (also used to group the dataset into coherent parts). |
| star_rating       | The 1-5 star rating of the review.                                                                             |
| helpful_votes     | Number of helpful votes.                                                                                       |
| total_votes       | Number of total votes the review received.                                                                     |
| vine              | Review was written as part of the Vine program.                                                                |
| verified_purchase | The review is on a verified purchase.                                                                          |
| review_headline   | The title of the review.                                                                                       |
| review_body       | The review text.                                                                                               |
| review_date       | The date the review was written.                                                                               |

## Data Cleaning

The following was done to clean up the data:

- Removed missing values from `review_body`
- Subset the columns `star_rating` and `review_body`
- Added in a new column `good_review` that converts the star rating to a binary class
- Removed duplicate reviews

## Text Cleaning

The following was done to clean up the text:

- Made text lowercase, removed text in square brackets, removed punctuation and removed words containing numbers
- Removed stopwords since they do not have any value in predicting sentiment
- Sampled 5000 reviews for analysis
- Split the data into 70% traning and 30% test sets

## Exploratory Data Analysis

Distribution of Good Reviews  |  Distribution of Number of Words
:-------------------------:|:-------------------------:
![alt text](https://github.com/nchin212/Sentiment_Analysis/blob/gh-pages/plots/bar3.png) |  ![alt text](https://github.com/nchin212/Sentiment_Analysis/blob/gh-pages/plots/bar2.png)

Most Frequent Words        |  Most Frequent Words for Positive Sentiment
:-------------------------:|:-------------------------:
![alt text](https://github.com/nchin212/Sentiment_Analysis/blob/gh-pages/plots/cloud1.png) |  ![alt text](https://github.com/nchin212/Sentiment_Analysis/blob/gh-pages/plots/cloud2.png)

## Tokenization

### CountVectorizer

CountVectorizer uses the Bag-of-words model(BoW) which is the simplest way of extracting features from text. BoW converts text into the matrix of occurrence of words within a document. This matrix is known as a Document-Term Matrix(DTM). In this case, we will create the DTM using CountVectorizer and only 1-gram models will be used.

### TfidfVectorizer

Rather than compute the frequency of words in a text, tf-idf, which stands for term frequency — inverse document frequency, computes how relevant a term is in a document. Take note that the following calculations are used for scikit-learn's TfidfVectorizer so how tf-idf is calculated in other cases may vary. The details are as follows:

TF (Term Frequency) - Measures the frequency of a word in a text, calculated by taking the number of times the word occurs in the text divided by the total number of words in the text.

IDF (Inverse Document Frequency) - Measures how much information the word provides, i.e., if it's common or rare across all text, calculated as follows: ln(n + 1 / df + 1) + 1, where n refers to the total number of texts and df (document frequency) refers to the number of texts that contain the word. IDF diminishes the weight of terms that occur very frequently in the document set and increases the weight of terms that occur rarely.

TF-IDF (Term Frequency — Inverse Document Frequency) - Reflects how important a word is to a document, calculated by taking TF multiplied by IDF

The tf-idf vectors are then normalized by the Euclidean norm, resulting in the output produced by TfidfVectorizer.

## Modelling

The following models were chosen to classify the reviews as positive or negative:

- Naive Bayes
- Logistic Regression
- Support Vector Machine (SVM)

## Results

|                        | Accuracy | Precision |   Recall | F1-score |
|-----------------------:|---------:|----------:|---------:|---------:|
|         Naive Bayes CV | 0.816667 |  0.822981 | 0.957543 | 0.885177 |
| Logistic Regression CV | 0.810667 |  0.848434 | 0.905149 | 0.875874 |
|                 SVM CV | 0.738000 |  0.738000 | 1.000000 | 0.849252 |
|         Naive Bayes TV | 0.740667 |  0.739973 | 1.000000 | 0.850557 |
| Logistic Regression TV | 0.801333 |  0.799408 | 0.975610 | 0.878763 |
|                 SVM TV | 0.738000 |  0.738000 | 1.000000 | 0.849252 |

Since the data is quite unbalanced, accuracy will not be a good indicator for which is the better model since the models will be biased towards the majority class. Thus, we should compare the f1-scores.

![alt text](https://github.com/nchin212/Sentiment_Analysis/blob/gh-pages/plots/bar4.png)


## Relevant Links

**Jupyter Notebook :** https://nchin212.github.io/Sentiment_Analysis/sentiment.html

**Portfolio :** https://nchin212.github.io/post/sentiment_analysis/
