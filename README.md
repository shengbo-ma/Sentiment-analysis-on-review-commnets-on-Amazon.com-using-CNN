# Sentiment-analysis-on-review-commnets-on-Amazon.com-using-CNN

Web mining excersize for deep learning using Convolutional Neural Network (CNN)

Amazon_review_500.csv is the input data for this excersize. This csv file has two columns as follows. The label
column provides polarity sentiment, either positive or negative.

---------------------------------------------
label text
2 I must admit that I'm addicted to "Version 2.0...
1 I think it's such a shame that an enormous tal...
2 The Sunsout No Room at The Inn Puzzle has oddl...
... ...
-----------------------------------------------


Q1: Train a CNN classification model
- Create a function sentiment_cnn() to detect sentiment as follows:
  - the input parameter is the full filename path to amazon_review_500.csv
  - convert the text into padded sequences of numbers (see Exercise 5.2)
  - hold 20% of the data for testing
  - carefully select hyperparameters: max number of words for embedding layer, input
  sentence length, filters, the number of filters, batch size, and epoch etc.
  - create a CNN model with the training data
  - print out accuracy, precision, recall calculated from testing data.
    - Your precision_macro, recall_macro, and accurracy should be all about 70%.
    - If your result is much lower than that (e.g. below 67%), you need to tune the
    hyperparameters
    - Also note that the label in the dataset is either 1 or 2. Your binary prediction out of
    CNN is either 0 or 1. Conversion is needed in order to compare predictions with
    actual labels
- This function has no return. Besides your code, also provide a pdf document showing the following
  - How you choose the hyperparameters
  - Screenshots of model trainning history
  - Testing accuracy, precision, recall

Q2 (Bonus) Improve the performance of CNN model
- Create a function improved_sentiment_cnn() to detect sentiment with improved accuracy
  - You still need to train a CNN model
  - You can apply different techniques, e.g.
  map words to pretrained word vectors
  e.g. from Google
  (https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?
  usp=sharing
  (https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?
  usp=sharing)) or
  e.g. from spacy package (https://spacy.io/usage/vectors-similarity
  (https://spacy.io/usage/vectors-similarity))
  e.g. create your own pretrained word vectors using other review
  documents you can find
  - add additional features etc.
  - Your taraget is to improve the accuracy by about 5% from the model you created in Q1.
  For fair comparison, make sure you use the same datasets for training/testing.
- This function has no return. Please provide a pdf document showing the following
  - Screenshots of model trainning history
  - Testing accuracy, precision, recall
  - Your analysis about
    - what technique contributes to the performance improvement
    - why this technique is useful
