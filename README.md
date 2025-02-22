# feature-engineering
NLP220 HW1 Report

Yifei Gan

1 Part A Naive Bayes:

![](Aspose.Words.8b2bf0e5-aae0-4e00-b96d-55b3daa4b4e2.001.png)

Figure 1: Distribution of Classes/Lables

1. Count Vectorizer

Count vectorizer converts text into a matrix of token counts, where each word or token is represented by its frequency in the document, capturing basic term occurrences.

Decision Tree:

1578 696 661 1565

Decision Tree Training Accuracy: 0.99996 Decision Tree Testing Accuracy: 0.69844



|Class|Precision|Recall F1-Score|Support|
| - | - | - | - |
|0 1|0\.70 0.69|<p>0.69 0.70</p><p>0.70 0.70</p>|2274 2226|
|Accuracy Macro avg|0\.70|<p>0\.70</p><p>0\.70 0.70</p>|4500|
|Weighted avg|0\.70|0\.70 0.70|4500|

Table 1: Classification report for the Decision Tree model with CountVectorizer.

1918 356 571 1655

Training Accuracy: 0.79286 Testing Accuracy: 0.794



|Class|Precision|Recall F1-Score Support|
| - | - | - |
|0|0\.77|0\.84 0.81 2274|
|1|0\.82|0\.74 0.78 2226|
|Accuracy||0\.79|
|Macro avg|0\.80|0\.79 0.79 4500|
|Weighted avg|0\.80|0\.79 0.79 4500|
|Table 2: Naive Bayes with CountVectorizer.|||
|<p>Support Vritual Machine:</p><p>1867 407</p>|||
|317 1909|||
|SVC Training Accuracy: 0.8532|||
|SVC Testing Accuracy: 0.83911|||
|Class|Precision|Recall F1-Score Support|
|0|0\.85|0\.82 0.84 2274|
|1|0\.82|0\.86 0.84 2226|
|Accuracy||0\.84|
|Macro avg|0\.84|0\.84 0.84 4500|
|Weighted avg|0\.84|0\.84 0.84 4500|

Table 3: SVC with CountVectorizer.

2. Count Vectorizer by characters

Count vectorizer by character is Similar to Count Vectorizer, but breaks text into sequences of characters rather than words, which helps capture word morphology or structural patterns.

Decision Tree:

1479 795 724 1502

Decision Tree Training Accuracy: 0.99996

Decision Tree Testing Accuracy: 0.66244 Decision Tree:

1593 681

|Class|Precision|Recall|F1-Score|Support|
| - | - | - | - | - |
|0 1|0\.67 0.65|0\.65 0.67|0\.66 0.66|2274 2226|

645 1581

Decision Tree Training Accuracy: 0.99996 Accuracy 0.66 Decision Tree Testing Accuracy: 0.70533

Macro avg 0.66 0.66 0.66 4500 ![](Aspose.Words.8b2bf0e5-aae0-4e00-b96d-55b3daa4b4e2.002.png)

Weighted avg 0.66 0.66 0.66 4500 Class Precision Recall F1-Score Support![](Aspose.Words.8b2bf0e5-aae0-4e00-b96d-55b3daa4b4e2.003.png)



|0 1|0\.71 0.70|<p>0.70 0.71</p><p>0.71 0.70</p>|2274 2226|
| - | - | - | - |
|Accuracy Macro avg Weighted avg|0\.71 0.71|<p>0\.71</p><p>0\.71 0.71</p><p>0\.71 0.71</p>|4500 4500|

Table 4: Decision Tree with CountVectorizer by charac- ters.

Naive Bayes:

1701 573

840 1386 Table 7: Decision Tree with TF-IDF.

Training Accuracy: 0.68188 Testing Accuracy: 0.686



|Class|Precision|Recall|F1-Score|Support|
| - | - | - | - | - |
|0 1|0\.67 0.71|0\.75 0.62|0\.71 0.66|2274 2226|

Naive Bayes:

1954 320 337 1889

Training Accuracy: 0.86403 Testing Accuracy: 0.854

Accuracy 0.69 ![](Aspose.Words.8b2bf0e5-aae0-4e00-b96d-55b3daa4b4e2.004.png) Macro avg 0.69 0.69 0.68 4500 Class Precision Recall F1-Score Support

Weighted avg 0.69 0.69 0.68 4500 0 0.85 0.86 0.86 2274![](Aspose.Words.8b2bf0e5-aae0-4e00-b96d-55b3daa4b4e2.005.png)![](Aspose.Words.8b2bf0e5-aae0-4e00-b96d-55b3daa4b4e2.006.png)

1 0.86 0.85 0.85 2226 Table 5: Naive Bayse with CountVectorizer by charac- ![](Aspose.Words.8b2bf0e5-aae0-4e00-b96d-55b3daa4b4e2.007.png) ters. Accuracy 0.85

Macro avg 0.85 0.85 0.85 4500 Weighted avg 0.85 0.85 0.85 4500

Support Virtual Machine: ![](Aspose.Words.8b2bf0e5-aae0-4e00-b96d-55b3daa4b4e2.008.png) Table 8: Naive Bayes with TF-IDF.

1830 444

441 1785 Support Virtual Machine:

SVC Training Accuracy: 0.823294 1976 298 SVC Testing Accuracy: 0.80333 267 1959![](Aspose.Words.8b2bf0e5-aae0-4e00-b96d-55b3daa4b4e2.009.png)

Class Precision Recall F1-Score Support SVC Training Accuracy: 0.9353

0 0.81 0.80 0.81 2274 SVC Testing Accuracy: 0.87444![](Aspose.Words.8b2bf0e5-aae0-4e00-b96d-55b3daa4b4e2.010.png)

1 0.80 0.80 0.80 2226 ![](Aspose.Words.8b2bf0e5-aae0-4e00-b96d-55b3daa4b4e2.011.png)

Accuracy 0.80 Class Precision Recall F1-Score Support Macro![](Aspose.Words.8b2bf0e5-aae0-4e00-b96d-55b3daa4b4e2.012.png)![](Aspose.Words.8b2bf0e5-aae0-4e00-b96d-55b3daa4b4e2.013.png) avg 0.80 0.80 0.80 4500 0 0.88 0.87 0.87 2274

Weighted avg 0.80 0.80 0.80 4500 1 0.87 0.88 0.87 2226![](Aspose.Words.8b2bf0e5-aae0-4e00-b96d-55b3daa4b4e2.014.png)![](Aspose.Words.8b2bf0e5-aae0-4e00-b96d-55b3daa4b4e2.015.png)

Accuracy 0.87

Table 6: SVC with CountVectorizer by characters. Macro avg 0.87 0.87 0.87 4500 Weighted avg 0.87 0.87 0.87 4500![](Aspose.Words.8b2bf0e5-aae0-4e00-b96d-55b3daa4b4e2.016.png)

3. TF-IDF Table 9: SVC with TF-IDF.

TF-IDF Weighs terms by their frequency in a

document relative to their frequency across the Here’s a summary of all the model performances. entire corpus, highlighting important terms while SVM is really good at handling complex data, es- downplaying common ones. pecially when there are lots of features like word

|Accuracy Rate|Count|Count by Character|TF-IDF|
| - | - | - | - |
|Decision Tree Naive Bayes SVM|0\.698 0.794 0.839|<p>0\.66 0.686</p><p>0\.80</p>|0\.705 0.854 0.87|

Table 10: Model Performance Metrics

counts or TF-IDF (a way of scoring important words). It looks for the best line (or boundary) that separates the data into classes, even if the data has many layers of complexity. It’s also good at not getting "confused" by noise in the data, so it can make better predictions overall.Naive Bayes works well with text data because it’s fast and easy. It assumes that all features are independent of each other, which isn’t always true, but often works well enough. It does a decent job but isn’t as flexible as SVM, which means it can miss some patterns in the data that SVM picks up on.Decision Trees struggle with more complicated data like text fea- tures (TF-IDF, word counts, etc.). They tend to "overthink" by splitting the data too much, which leads to poor predictions when you test it on new data. Also, they don’t handle a lot of features as well as SVM, so they end up performing the worst in this case.

2 Part B

I’ve used SVC, naive bayes, logistic regression, random forest classifier, and K-Nearest Neighbors. The text data was vectorized using CountVectorizer with an ngram range of (1, 2). This means that both unigrams (single words) and bigrams (pairs of consecutive words) were extracted as features. The Support Vector Classifier works by finding the optimal hyperplane that separates classes in the feature space. SVC is particularly effective in high-dimensional spaces, making it well-suited for text classification, where feature sets can be large due to the nature of textual data. The Naive Bayes classifier is a probabilistic model based on Bayes’ Theorem with the assumption that features are conditionally independent. Logistic Regression is a linear model used for binary classification tasks. It predicts the probability of a sample belonging to a particular class and assigns it based on a decision threshold. The Random Forest Classifier is an ensemble learning method that constructs multiple decision trees during training and outputs the mode of their predictions. It is robust to overfitting and provides high accuracy but can struggle with

sparse, high-dimensional data like text because of its reliance on decision trees, which may not handle such data effectively. K-Nearest Neighbors is a non-parametric, distance-based algorithm that classifies samples based on the majority class of their nearest neighbors. While KNN is conceptually simple and can be effective for some types of classification, it is computationally expensive for large datasets and performs poorly when faced with high-dimensional data like text, due to the curse of dimensionality.

To locate which measure of hyperparameters is the most accurate, optuna has been set and used in order to help.

1. Naive Bayes

alpha = trial.suggest\_loguniform( ' alpha ' , 1e-4, 1e2)

Best hyperparameters for Naive Bayes: alpha: 0.08561959327899799.

Naive Bayes Accuracy with Optuna: 0.8872.

2. Logistic Regression

C = trial.suggest\_loguniform( ' C' , 1e-4, 1e2)

solver = trial.suggest\_categorical( ' solver ' , [ ' liblinear ' , ' saga ' ])

Best hyperparameters for Logistic Regression: C: 0.07546343057088854; solver: liblinear.

Logistic Regression Accuracy with Optuna: 0.9060.

3. Random Forest Classifier

n\_estimators = trial.suggest\_int( ' n\_estimators ' , 10, 200) max\_depth = trial.suggest\_int( ' max\_depth ' , 2, 32)

min\_samples\_split = trial.suggest\_int( ' min\_samples\_split ' , 2, 10) min\_samples\_leaf = trial.suggest\_int( ' min\_samples\_leaf ' , 1, 10)

Best hyperparameters for Random For- est: n\_estimators: 179, max\_depth: 30; min\_samples\_split: 3; min\_samples\_leaf: 2.

Random Forest Accuracy with Optuna: 0.8772.

4. K-Nearest Neighbors

n\_neighbors = trial.suggest\_int( ' n\_neighbors ' , 1, 30) weights = trial.suggest\_categorical( ' weights ' ,

[ ' uniform ' , ' distance ' ])

algorithm = trial.suggest\_categorical( ' algorithm ' ,

[ ' auto ' , ' ball\_tree ' , ' kd\_tree ' , ' brute ' ])

Best hyperparameters for K-Nearest Neighbors: n\_neighbors: 30; weights: uniform; algorithm: brute.

K-Nearest Neighbors Accuracy with Optuna: 0.6472.

5. SVC

I used both SVC and Linear SVC, which resulted in completely differnt time of running the code. For LinearSVC, it only cost me several seconds, however, the regular SVC kernel linear took me more than 1100 hours to find the best parameters. Additionally, the accuracy rate for SVC is even lower than LinearSVC.

The dataset is then split into 80% train and 20% test parts using train\_test\_split.

train\_sentences , test\_sentences , train\_labels , test\_labels =train\_test\_split(

df[ ' review/text ' ].values,

encoded\_labels ,

stratify=encoded\_labels , test\_size=0.15)

After this I use CountVectorizer to get the fre- quency of each word appearing in the training set. I store them in a dictionary called ‘word\_counts’. All the unique words in the corpus are stored in ‘vocab.’

vec = CountVectorizer(max\_features = 3000)

X = vec.fit\_transform(train\_sentences)

vocab = vec.get\_feature\_names\_out()

1. SVC X = X.toarray()

word\_counts = {}

C = trial.suggest\_loguniform( ' C' , 1e-4, 1e2) for l in range(2):

kernel = trial.suggest\_categorical( ' kernel ' , [ ' linear ' , word\_counts[l] = defaultdict(lambda: 0)

' poly ' , ' rbf ' , ' sigmoid ' ]) for i in range(X.shape[0]):

gamma = trial.suggest\_categorical( ' gamma' , [ ' scale ' , ' auto ' ])l = train\_labels[i]

for j in range(len(vocab)):

Best hyperparameters for Support Vector Classifier: word\_counts[l][vocab[j]] += X[i][j]

C: 0.007482667477667498; kernel: linear; gamma: Then the fit and predict function is defined. The scale. ‘fit’ function takes reviews and labels values to be Support Vector Classifier Accuracy with Optuna: fitted on and returns the number of reviews with 0.8992. each label and the apriori conditional probabilities.

2. LinearSVC

C = trial.suggest\_loguniform( ' C' , 1e-4, 1e2)

Best hyperparameters for Support Vector Classifier: C: 0.002863754188529008.

Support Vector Classifier Accuracy with Optuna: 0.9028.

3 Part C

After loading the data, I preprocess the data, elim- inating all the review\_score = 3.0, and assign bi- nary\_labels for them.

df = df[df[ ' review/score ' ] != 3]

df[ ' binary\_label ' ] = df[ ' review/score ' ].apply

(lambda x: ' positive ' if x > 3 else ' negative ' )

Following I perform lemmatization using wordnet in nltk in order to help saving unnecessary compu- tational overhead in deciphering entire words since most words’ meanings are well-expressed by their separate lemmas. The text is first broken into in- dividual tokens using the WhitespaceTokenizer() from nltk.

import nltk

nltk.download( ' wordnet ' )

w\_tokenizer = nltk.tokenize.WhitespaceTokenizer() lemmatizer = nltk.stem.WordNetLemmatizer()

def lemmatize\_text(text):

st = ""

for w in w\_tokenizer.tokenize(text):

st = st + lemmatizer.lemmatize(w) + " " return st

df[ ' review/text ' ] = df[ ' review/text ' ]

.apply(lemmatize\_text)

def fit(x, y, labels):

n\_label\_items = {}

log\_label\_priors = {}

n = len(x)

grouped\_data = group\_by\_label(x, y, labels) for l, data in grouped\_data.items():

n\_label\_items[l] = len(data)      log\_label\_priors[l] = math.log (n\_label\_items[l] / n)

return n\_label\_items , log\_label\_priors

def predict(n\_label\_items , vocab, word\_counts ,

log\_label\_priors , labels, x):

result = []

for text in x:

label\_scores = {l: log\_label\_priors[l]

for l in labels}

words = set(w\_tokenizer.tokenize(text))

for word in words:

if word not in vocab: continue

for l in labels:

log\_w\_given\_l = laplace\_smoothing

(n\_label\_items , vocab, word\_counts , word, l) label\_scores[l] += log\_w\_given\_l result.append(max(label\_scores , key=label\_scores.get))

return result

At last get the accuracy of the model.

labels = [0,1]

n\_label\_items , log\_label\_priors = fit

(train\_sentences ,train\_labels ,labels) pred = predict(n\_label\_items , vocab,

word\_counts , log\_label\_priors , labels, test\_sentences) print("Accuracy of prediction on test set :

", accuracy\_score(test\_labels ,pred))

The Bayes Therom:

P (B|A) ·P (A) P (A|B) =

P (B)

where: P (A|B) is the probability of event A occur- ring given that B is true. P (B|A) is the probability of event B occurring given that A is true. P (A) is

the probability of event A. P (B) is the probability of event B.

For a given X{a1,a2,a3,...,an} to be classified, calculate the probability of each X occurring under the condition yi that this item appears. The item is assigned to the class with the highest P (yi | X).
