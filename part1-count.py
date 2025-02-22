
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import torch
import torch.utils
import torch.utils.data
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("small_books_rating.csv")
df = df[df['review/score'] != 3]
df['binary_label'] = df['review/score'].apply(lambda x: 'positive' if x > 3 else 'negative')
df.head()

x = df["review/text"]
y = df["binary_label"]
countVectorizer = CountVectorizer(max_features=1000)
x_train = countVectorizer.fit_transform(x)
x_test = countVectorizer.transform(x)
x_train, x_test, y_train, y_test = train_test_split(x_train, y, test_size=0.15, random_state=42)

y_train = y_train.apply(lambda x: 1 if x == 'positive' else 0)
y_test = y_test.apply(lambda x: 1 if x == 'positive' else 0)

x_train_vec = torch.tensor(x_train.toarray(), dtype=torch.float32)
y_train_vec = torch.tensor(y_train.to_numpy(), dtype=torch.float32)
x_test_vec = torch.tensor(x_test.toarray(), dtype=torch.float32)
y_test_vec = torch.tensor(y_test.to_numpy(), dtype=torch.float32)

# Decision Tree.
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(x_train, y_train)

y_pred_train_dt = dt_model.predict(x_train)
y_pred_test_dt = dt_model.predict(x_test)

train_accuracy_dt = accuracy_score(y_train, y_pred_train_dt)
test_accuracy_dt = accuracy_score(y_test, y_pred_test_dt)

print(f"Decision Tree Training Accuracy: {train_accuracy_dt}")
print(f"Decision Tree Testing Accuracy: {test_accuracy_dt}")
print("Decision Tree Classification Report:\n", classification_report(y_test, y_pred_test_dt))

# Compute the macro F1 score
macro_f1 = f1_score(y_test, y_pred_test_dt, average='macro')
print(f"Macro F1 Score: {macro_f1}")

# Print the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_test_dt)
print("Confusion Matrix:\n", conf_matrix)
#
#
#
# Naive Bayes.
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

nb_model = MultinomialNB()

nb_model.fit(x_train, y_train)

y_pred_train = nb_model.predict(x_train)
y_pred_test = nb_model.predict(x_test)

train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

print(f"Training Accuracy: {train_accuracy}")
print(f"Testing Accuracy: {test_accuracy}")
print("Classification Report:\n", classification_report(y_test, y_pred_test))

# Compute the macro F1 score
macro_f1 = f1_score(y_test, y_pred_test, average='macro')
print(f"Macro F1 Score: {macro_f1}")

# Print the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_test)
print("Confusion Matrix:\n", conf_matrix)
#
#
#
# Support Vector Machine.
svc_model = LinearSVC()

svc_model.fit(x_train, y_train)

y_pred_train_svc = svc_model.predict(x_train)
y_pred_test_svc = svc_model.predict(x_test)

train_accuracy_svc = accuracy_score(y_train, y_pred_train_svc)
test_accuracy_svc = accuracy_score(y_test, y_pred_test_svc)

print(f"SVC Training Accuracy: {train_accuracy_svc}")
print(f"SVC Testing Accuracy: {test_accuracy_svc}")
print("SVC Classification Report:\n", classification_report(y_test, y_pred_test_svc))

# Compute the macro F1 score
macro_f1 = f1_score(y_test, y_pred_test_svc, average='macro')
print(f"Macro F1 Score: {macro_f1}")

# Print the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_test_svc)
print("Confusion Matrix:\n", conf_matrix)



