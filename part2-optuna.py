import pandas as pd
from sklearn.calibration import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import optuna

df = pd.read_csv('imdb_reviews.csv')

x = df['review']
y = df['sentiment']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

vectorizer = CountVectorizer(ngram_range=(1, 2))
x_train_transformed = vectorizer.fit_transform(x_train)
x_test_transformed = vectorizer.transform(x_test)

#SVC
#!!WARNING this may take over 1000 HOURS to run!!
def svc_objective(trial):
    C = trial.suggest_loguniform('C', 1e-4, 1e2)
    kernel = trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
    gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])
    
    svc_model = SVC(C=C, kernel=kernel, gamma=gamma)
    svc_model.fit(x_train_transformed, y_train)
    
    y_val_pred_svc = svc_model.predict(x_test_transformed)
    accuracy = accuracy_score(y_test, y_val_pred_svc)
    
    return accuracy

svc_study = optuna.create_study(direction='maximize')
svc_study.optimize(svc_objective, n_trials=50)

best_svc_params = svc_study.best_params
print(f'Best hyperparameters for Support Vector Classifier: {best_svc_params}')

svc_model = SVC(**best_svc_params)
svc_model.fit(x_train_transformed, y_train)

y_val_pred_svc = svc_model.predict(x_test_transformed)

svc_accuracy = accuracy_score(y_test, y_val_pred_svc)
print(f'Support Vector Classifier Accuracy with Optuna: {svc_accuracy:.4f}')
#
#
#
#LinearSVC
def svc_objective(trial):
    C = trial.suggest_loguniform('C', 1e-4, 1e2)
    svc_model = LinearSVC(C=C)
    
    svc_model.fit(x_train_transformed, y_train)
    y_val_pred_svc = svc_model.predict(x_test_transformed)

    accuracy = accuracy_score(y_test, y_val_pred_svc)
    
    return accuracy

svc_study = optuna.create_study(direction='maximize')
svc_study.optimize(svc_objective, n_trials=50)

best_svc_params = svc_study.best_params
print(f'Best hyperparameters for Support Vector Classifier: {best_svc_params}')

svc_model = LinearSVC(**best_svc_params)
svc_model.fit(x_train_transformed, y_train)

y_val_pred_svc = svc_model.predict(x_test_transformed)

svc_accuracy = accuracy_score(y_test, y_val_pred_svc)
print(f'Linear Support Vector Classifier Accuracy with Optuna: {svc_accuracy:.4f}')
#
#
#
#Naive Bayes
def nb_objective(trial):
    alpha = trial.suggest_loguniform('alpha', 1e-4, 1e2)
    
    nb_model = MultinomialNB(alpha=alpha)
    nb_model.fit(x_train_transformed, y_train)
    
    y_val_pred_nb = nb_model.predict(x_test_transformed)
    
    accuracy = accuracy_score(y_test, y_val_pred_nb)
    
    return accuracy

nb_study = optuna.create_study(direction='maximize')
nb_study.optimize(nb_objective, n_trials=50)

best_nb_params = nb_study.best_params
print(f'Best hyperparameters for Naive Bayes: {best_nb_params}')

nb_model = MultinomialNB(**best_nb_params)
nb_model.fit(x_train_transformed, y_train)

y_val_pred_nb = nb_model.predict(x_test_transformed)

nb_accuracy = accuracy_score(y_test, y_val_pred_nb)
print(f'Naive Bayes Accuracy with Optuna: {nb_accuracy:.4f}')
#
#
#
#Logistic Regression
def objective(trial):
    C = trial.suggest_loguniform('C', 1e-4, 1e2)
    solver = trial.suggest_categorical('solver', ['liblinear', 'saga'])
    
    logistic_model = LogisticRegression(C=C, solver=solver, max_iter=1000)
    logistic_model.fit(x_train_transformed, y_train)
    
    y_val_pred_logistic = logistic_model.predict(x_test_transformed)

    accuracy = accuracy_score(y_test, y_val_pred_logistic)
    
    return accuracy

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

best_params = study.best_params
print(f'Best hyperparameters: {best_params}')

logistic_model = LogisticRegression(**best_params, max_iter=1000)
logistic_model.fit(x_train_transformed, y_train)

y_val_pred_logistic = logistic_model.predict(x_test_transformed)

logistic_accuracy = accuracy_score(y_test, y_val_pred_logistic)
print(f'Logistic Regression Accuracy with Optuna: {logistic_accuracy:.4f}')
#
#
#
#Random Forest
def rf_objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 10, 200)
    max_depth = trial.suggest_int('max_depth', 2, 32)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
    
    rf_model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )
    
    rf_model.fit(x_train_transformed, y_train)
    
    y_val_pred_rf = rf_model.predict(x_test_transformed)
    
    accuracy = accuracy_score(y_test, y_val_pred_rf)
    
    return accuracy

rf_study = optuna.create_study(direction='maximize')
rf_study.optimize(rf_objective, n_trials=50)

best_rf_params = rf_study.best_params
print(f'Best hyperparameters for Random Forest: {best_rf_params}')

rf_model = RandomForestClassifier(**best_rf_params, random_state=42)
rf_model.fit(x_train_transformed, y_train)

y_val_pred_rf = rf_model.predict(x_test_transformed)

rf_accuracy = accuracy_score(y_test, y_val_pred_rf)
print(f'Random Forest Accuracy with Optuna: {rf_accuracy:.4f}')
#
#
#
# K-Nearest Neighbors
def knn_objective(trial):
    n_neighbors = trial.suggest_int('n_neighbors', 1, 30)
    weights = trial.suggest_categorical('weights', ['uniform', 'distance'])
    algorithm = trial.suggest_categorical('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute'])
    
    knn_model = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=weights,
        algorithm=algorithm
    )
    
    knn_model.fit(x_train_transformed, y_train)
    y_val_pred_knn = knn_model.predict(x_test_transformed)
    
    accuracy = accuracy_score(y_test, y_val_pred_knn)
    
    return accuracy

knn_study = optuna.create_study(direction='maximize')
knn_study.optimize(knn_objective, n_trials=50)

best_knn_params = knn_study.best_params
print(f'Best hyperparameters for K-Nearest Neighbors: {best_knn_params}')

knn_model = KNeighborsClassifier(**best_knn_params)
knn_model.fit(x_train_transformed, y_train)

y_val_pred_knn = knn_model.predict(x_test_transformed)

knn_accuracy = accuracy_score(y_test, y_val_pred_knn)
print(f'K-Nearest Neighbors Accuracy with Optuna: {knn_accuracy:.4f}')


