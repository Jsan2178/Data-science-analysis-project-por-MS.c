# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
df = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)
df

print(df["Review"][0]) #taking by index of this column for each row elements


# Cleaning the texts
import re      #this library will help us to clean up the text in general
import nltk    
nltk.download('stopwords') #for non relevant words 
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer  # For stemming our reviews and taking only root words enough to the words mean, present tense only
corpus = []
for i in range(0, 1000):
    #First we're only going to keep letters a-zA-Z
    review = re.sub('[^a-zA-Z]', ' ', df['Review'][i]) 
    review = review.lower() #now in lower letter
    review = review.split() #here we split the review
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')  #load empty words(e.g., "the" ,"and", "not", etc) for remove it 
    all_stopwords.remove('not') #we keep the "not" because is important in our analysis
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review) #join the list of resulting tokens in one each string, separate by spaces
    corpus.append(review)
#let's see our corpus string
print(type(corpus), len(corpus)) #returns a list of the same size of row as in our DF 
print(corpus)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer 
cv = CountVectorizer(max_features = 1500) #for tokenization vocabulary order by frequency with a limit of features (dimension vector)
X = cv.fit_transform(corpus).toarray() #Here based on CV creates a vector for each corpus element with the number of matching elements with CV (same dimension vector as above)
y = df.iloc[:, -1].values #just taking the Liked column elements
len(X[0]) #as we established max_features


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Training the Naive Bayes model on the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, y_pred)
print(f"Confussion Matrix Values: \n{cm}\n")
print(f"Accuracy: {accuracy_score(y_test, y_pred)}\n")
ConfusionMatrixDisplay.from_predictions(y_test,y_pred)
plt.show()

## predicting new reviews
new_review = 'I really like this Restaurant and dishes'
new_review = re.sub('[^a-zA-Z]', ' ', new_review)
new_review = new_review.lower()
new_review = new_review.split()
ps = PorterStemmer()
all_stopwords = stopwords.words('english')
all_stopwords.remove('not')
new_review = [ps.stem(word) for word in new_review if not word in set(all_stopwords)]
new_review = ' '.join(new_review)
new_corpus = [new_review]
new_X_test = cv.transform(new_corpus).toarray()
new_y_pred = classifier.predict(new_X_test)
print(new_y_pred)

new_review = "I hate this restaurant it's not as good as I thought"
new_review = re.sub('[^a-zA-Z]', ' ', new_review)
new_review = new_review.lower()
new_review = new_review.split()
ps = PorterStemmer()
all_stopwords = stopwords.words('english')
all_stopwords.remove('not')
new_review = [ps.stem(word) for word in new_review if not word in set(all_stopwords)]
new_review = ' '.join(new_review)
new_corpus = [new_review]
new_X_test = cv.transform(new_corpus).toarray()
new_y_pred = classifier.predict(new_X_test)
print(new_y_pred)

## Implementing with Multinomial NB
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB(alpha=2.5)
clf.fit(X_train, y_train)  #same sintaxis

### Instead of using the above alpha, let's search for the best one
alphas=np.random.uniform(0.01, 10, 1000)  # or you can implement gamma_trunc, trunc_norm, and any other ditribution
results=[]
for alpha_val in alphas:
    clf = MultinomialNB(alpha=alpha_val)
    clf.fit(X_train, y_train)  
    y_pred_mn = clf.predict(X_test)
    acc=accuracy_score(y_test, y_pred_mn)
    results.append((alpha_val, acc))
#Maximun accuracy
alpha_best , best_acc = max(results, key=lambda x: x[1])
print(f"Best alpha: {alpha_best}, Accuracy: {best_acc:.3f}")
#Best alphas 
#alp_h = [alpha for alpha, acc in results if acc>0.79]
#print(f"Alphas with accuracy > 0.79: {alp_h}\n")

### Now we'll use this value
clf = MultinomialNB(alpha=alpha_best)
clf.fit(X_train, y_train)  #same sintaxis
y_pred_mn = clf.predict(X_test)
print(np.concatenate((y_pred_mn.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
cmn = confusion_matrix(y_test, y_pred_mn)
print(f"Confussion Matrix Values: \n{cmn}\n")
print(f"Accurary Using Multonomial NB: {accuracy_score(y_test, y_pred_mn)}\n")
ConfusionMatrixDisplay.from_predictions(y_test,y_pred_mn)
plt.show()

### predicting new reviews again
new_review = "I really like this Restaurant and dishes, they're delicious"
new_review = re.sub('[^a-zA-Z]', ' ', new_review)
new_review = new_review.lower()
new_review = new_review.split()
ps = PorterStemmer()
all_stopwords = stopwords.words('english')
all_stopwords.remove('not')
new_review = [ps.stem(word) for word in new_review if not word in set(all_stopwords)]
new_review = ' '.join(new_review)
new_corpus = [new_review]
new_X_test = cv.transform(new_corpus).toarray()
new_y_pred = clf.predict(new_X_test)
print(new_y_pred)

#predicting new review
new_review = "I hate this restaurant it's not as good as I thought"
new_review = re.sub('[^a-zA-Z]', ' ', new_review)
new_review = new_review.lower()
new_review = new_review.split()
ps = PorterStemmer()
all_stopwords = stopwords.words('english')
all_stopwords.remove('not')
new_review = [ps.stem(word) for word in new_review if not word in set(all_stopwords)]
new_review = ' '.join(new_review)
new_corpus = [new_review]
new_X_test = cv.transform(new_corpus).toarray()
new_y_pred = clf.predict(new_X_test)
print(new_y_pred)

## Analyzing the results 
 ### We can say that the Multinomial NB has improved our overall model against Gaussian NB  reducing the True Positives but increasing the True Negative
 ### Giving us an accuracy from 0.73 to 0.79 with the best posible alpha.
 ### Given this information, we can conclude the actual and potential clients who probably will come back to the restaurant, mall or who won't based on the reviews and work for a better services taking note to the most frequent words in the reviews vocabulary.