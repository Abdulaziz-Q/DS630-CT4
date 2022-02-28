# Import packages needed for this critical thinking 4
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


"""
We work on a partial dataset with only 4 categories out of 20 available in the datasets 
in order to get quicker run times for this first example:

"""
categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']



# The list of files corresponding to the following categories can now be loaded:
twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)



"""
The CountVectorizer includes the text preprocessing, the tokenizing and the filtering of stopwords.
It creates a features dictionary and converts documents into vectors:
"""
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)



"""
In order to eliminate these possible variations, the number of occurrences of each word in a document is sufficient 
to be separated by the total number of terms in the document: tf for Term Frequencies is called these new features.
"""
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)



# With our features, we can now train a naive bayes  classifier in to attempt to predict a post group.
clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)



"""
In order to try to predict the result of a new paper, we need to extract the characteristics with almost the same extraction 
chain as before. The difference is that we call transform on the transformers rather than fit_transform as they already fit 
the collection of training:
"""
docs_new = ['God is love', 'OpenGL on the GPU is fast']
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

predicted = clf.predict(X_new_tfidf)

for doc, category in zip(docs_new, predicted):
    print('%r => %s' % (doc, twenty_train.target_names[category]))



"""
In order to make the vectorizer => transformer => classifier easier to work with, scikit-learn provides 
a Pipeline class that behaves like a compound classifier:
"""
text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB()), ])



"""
Arbitrary names vect, tfidf and clf (classifier). We'll use them to scan the grid for the following relevant hyperparameters.
With one single control we can now train the model:
"""
text_clf.fit(twenty_train.data, twenty_train.target)



# Evaluate the model's predictive accuracy
twenty_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)
docs_test = twenty_test.data
predicted = text_clf.predict(docs_test)
np.mean(predicted == twenty_test.target)