from sklearn.feature_extraction.text import CountVectorizer

corpus = [
    'This is a sample text.',
    'This text is another example text.',
    'This is just another text.'
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)

print(X.toarray())

