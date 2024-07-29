# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

# Load libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from pandas import read_csv
import ssl
import certifi
from urllib import request
from io import StringIO

# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']

# Create an SSL context using certifi's certificate bundle
context = ssl.create_default_context(cafile=certifi.where())

# Download the CSV file with a custom SSL context
response = request.urlopen(url, context=context)

# Read the CSV file into a pandas DataFrame
dataset = read_csv(StringIO(response.read().decode('utf-8')), names=names)

# shape
print(dataset.shape)
# head
print(dataset.head(20))
# descriptions
print(dataset.describe())
# class distribution
print(dataset.groupby('class').size())

#histograms of data
dataset.hist()
plt.show()

#multivariate analysis with scatter_matrix
...
# scatter plot matrix
scatter_matrix(dataset)
plt.show()

#creating dataset into training and validation sets
ary = dataset.values
X = ary[:,0:4]
y = ary[:,4]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=.2, random_state=1)


...
#making the models
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
# evaluate each model
results = []
names = []
for name, model in models:
	kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
#print the name, mean and standard deviation of each model
	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

#from the k fold cross validation we see that SVM (support vector machines) gave the most
# accurate scores so we will use it to make predictions

#making predictions on validation set
model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)


# evaluate predictions
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
