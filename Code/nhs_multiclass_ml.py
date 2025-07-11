import sys
import pandas as pd
import numpy as np
import csv
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import scipy.stats

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import label_binarize

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier

from sklearn.metrics import accuracy_score, classification_report, f1_score, recall_score, precision_score, average_precision_score, precision_recall_curve
import matplotlib.pyplot as plt
import scikitplot as skplt

stop_words = set(stopwords.words('english'))

df_train = pd.read_excel("final_trainset.xlsx")
df_test = pd.read_excel("final_testset.xlsx")
df_train['ABSTRACT'] = df_train['ABSTRACT'].apply(str.lower)
df_test['ABSTRACT'] = df_test['ABSTRACT'].apply(str.lower)
df_train['filtered_abstract'] = df_train['ABSTRACT'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in stop_words]))
df_test['filtered_abstract'] = df_test['ABSTRACT'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in stop_words]))
df_train['Correct_Label'] = df_train["Correct_Label"] + 1
df_test['Correct_Label'] = df_test["Correct_Label"] + 1

def init_model(model_type, hyperparams={}):
    model = None
    model_name = ''
    if model_type == 'nb':
        model = MultinomialNB()
        model_name = 'Naive Bayes'
    elif model_type == 'lr':
        model = LogisticRegression(max_iter=1000, C=hyperparams['c_val'])
        model_name = 'Logistic Regression'
    elif model_type == 'svm':
        model = LinearSVC()
        model_name = 'Support Vector Machine'
    elif model_type == 'knn':
        model = KNeighborsClassifier(n_neighbors=hyperparams['n_neighbors'])
        model_name = 'K-Nearest Neighbors'
    elif model_type == 'rf':
        model = RandomForestClassifier(n_estimators=hyperparams['n_est'])
        model_name = 'Random Forest'
    elif model_type == 'ada':
        model = AdaBoostClassifier(n_estimators=hyperparams['n_est'], learning_rate=hyperparams['learn_rate'])
        model_name = 'Ada Boost'
    elif model_type == 'gb':
        model = GradientBoostingClassifier(n_estimators=hyperparams['n_est'], learning_rate=hyperparams['learn_rate'])
        model_name = 'Gradient Boost'
    return model, model_name

def output_metrics(metric_dict, model_name, filename):
    columns = ['TrainingAccuracy', 'TestAccuracy', 'Precision', 'Recall', 'F1', 'AvgPrecisionScore']
    out_df = pd.DataFrame(metric_dict, columns=columns)
    out_df.to_csv(f'{model_name}-{filename}.csv', index=False)

def conf_interval(confidence, y_shape, metric):
    z_value = scipy.stats.norm.ppf((1 + confidence) / 2.0)
    ci_length = z_value * np.sqrt((metric * (1 - metric)) / y_shape)
    return ci_length

def multi_matrix(y_test, y_pred, model_type, vector_type):
    fig = plt.figure(figsize=(15,6))
    ax1 = fig.add_subplot(121)
    skplt.metrics.plot_confusion_matrix(y_test, y_pred,
                                    title="Multiclass Confusion Matrix",
                                    ax=ax1)
    plt.savefig(model_type + '_' + vector_type + '_multimat.png')

def multi_model(model_type, vectortype, hyperparams):

    abstracts = df_train['ABSTRACT'].values
    test_abstracts = df_test['ABSTRACT'].values
    if vectortype == "tfidf":
        vectorizer = TfidfVectorizer()
    elif vectortype == "countvec":
        vectorizer = CountVectorizer()
        abstracts = df_train['filtered_abstract'].values
        test_abstracts = df_test['filtered_abstract'].values
    
    vec_abstracts = np.array(vectorizer.fit_transform(abstracts).todense())
    X_train = vec_abstracts
    y_train = df_train['Correct_Label'].values

    model, model_name = init_model(model_type, hyperparams)
        
    model.fit(X_train, y_train)

    trainaccuracy = model.score(X_train, y_train)
    print('Training Accuracy:', trainaccuracy)

    vec_test = vectorizer.transform(test_abstracts)
    X_test = np.array(vec_test.todense())
    y_test = df_test['Correct_Label'].values
    y_pred = model.predict(X_test)

    if model_type == 'svm':
        y_score = model.decision_function(X_test)
    else:
        y_score = model.predict_proba(X_test)
    
    y_bin = label_binarize(y_test, classes=[0, 1, 2, 3])

    testaccuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    precision = precision_score(y_test, y_pred, average="macro")
    recall = recall_score(y_test, y_pred, average="macro")
    aps = average_precision_score(y_bin, y_score, average="macro")

    metrics = {'TrainingAccuracy': trainaccuracy, 'TestAccuracy': testaccuracy, 
                'Precision': precision, 'Recall': recall, 'F1': f1, 'AvgPrecisionScore': aps}
    conf_scores = {}
    for metric in metrics:
        conf_scores[metric] = conf_interval(0.95, y_test.shape[0], metrics[metric])
    
    print('Test Accuracy:', testaccuracy)
    print('F1 score:', f1)
    print('Precision:', precision)
    print('Recall:', recall)
    print('Average Precision Score:', aps)
    
    metric_dict = [metrics, conf_scores]
    output_metrics(metric_dict, model_name, f'FinalMultiClass-n50-{vectortype}')
    multi_matrix(y_test, y_pred, model_type, vectortype)

    return y_test, y_pred, y_score

if __name__ ==  "__main__":
    model_type = 'lr' # 'lr', 'svm', 'knn', 'rf', 'ada', or 'gb'
    vector_type = 'countvec' # countvec or tfidf
    hyperparams = {} # e.g. {'n_est': 50, 'learn_rate': 1} see init_model function
    multi_model(model_type, vector_type, hyperparams)

