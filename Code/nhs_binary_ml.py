import sys
import pandas as pd
import numpy as np
import csv
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import scipy.stats

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, classification_report, f1_score, recall_score, precision_score, average_precision_score, precision_recall_curve
import matplotlib.pyplot as plt
import scikitplot as skplt


stop_words = set(stopwords.words('english'))
df_bin = pd.read_excel("final_trainset.xlsx")
df_bin['binary2_label'].value_counts()
df_test = pd.read_excel("final_testset.xlsx")
df_bin['ABSTRACT'] = df_bin['ABSTRACT'].apply(str.lower)
df_test['ABSTRACT'] = df_test['ABSTRACT'].apply(str.lower)
df_bin['filtered_abstract'] = df_bin['ABSTRACT'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in stop_words]))
df_test['filtered_abstract'] = df_test['ABSTRACT'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in stop_words]))


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
        model = GradientBoostingClassifier(n_estimators=hyperparams['n_est'])
        model_name = 'Gradient Boost'
    return model, model_name

def output_metrics(metric_dict, model_name, classify_type):
    columns = ['TrainingAccuracy', 'TestAccuracy', 'Precision', 'Recall', 'F1', 'AvgPrecisionScore']
    out_df = pd.DataFrame(metric_dict, columns=columns)
    out_df.to_csv(f'{model_name}-{classify_type}.csv', index=False)

def conf_interval(confidence, y_shape, metric):
    z_value = scipy.stats.norm.ppf((1 + confidence) / 2.0)
    ci_length = z_value * np.sqrt((metric * (1 - metric)) / y_shape)
    return ci_length

def bin_matrix(y_test, y_pred, model_type, vector_type):
    fig = plt.figure(figsize=(15,6))
    ax1 = fig.add_subplot(121)
    skplt.metrics.plot_confusion_matrix(y_test, y_pred,
                                    title="Binary Confusion Matrix",
                                    ax=ax1)
    plt.savefig(model_type + '_' + vector_type + '_binmat.png')

def bin_model(model_type, vectortype, hyperparams):
    abstracts = df_bin['ABSTRACT'].values
    test_abstracts = df_test['ABSTRACT'].values
    if vectortype == "tfidf":
        vectorizer = TfidfVectorizer()
    elif vectortype == "countvec":
        vectorizer = CountVectorizer()
        abstracts = df_bin['filtered_abstract'].values
        test_abstracts = df_test['filtered_abstract'].values
    
    vec_abstracts = np.array(vectorizer.fit_transform(abstracts).todense())
    
    X_train = vec_abstracts
    y_train = df_bin['binary2_label'].values
    
    # initiate model and train on full trainset
    model, model_name = init_model(model_type, hyperparams)
    model.fit(X_train, y_train)

    trainaccuracy = model.score(X_train, y_train)
    print('Training Accuracy:', trainaccuracy)

    # test on hold out test set
    vec_test = vectorizer.transform(test_abstracts)
    X_test = np.array(vec_test.todense())
    y_test = df_test['binary2_label'].values
    y_pred = model.predict(X_test)

    if model_type == 'svm':
        y_score = model.decision_function(X_test)
    else:
        y_score = model.predict_proba(X_test)[:, 1]

    testaccuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    aps = average_precision_score(y_test, y_score)

    metrics = {'TrainingAccuracy': trainaccuracy, 'TestAccuracy': testaccuracy, 
                        'Precision': precision, 'Recall': recall, 'F1': f1, 'AvgPrecisionScore': aps}
    
    print('Test Accuracy:', testaccuracy)
    print('F1 score:', f1)
    print('Precision:', precision)
    print('Recall:', recall)
    print('Average Precision Score:', aps)
    
    precision, recall, _ = precision_recall_curve(y_test, y_score)

    plt.plot(recall, precision, color='blue',
             label=r'Precision-Recall (AUC = %0.2f)' % (aps),
             lw=1)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{model_name} PR Curve')
    plt.legend(loc="lower right")
    plt.savefig(f'{model_name}-PRCurve-Bin-{vectortype}.png')

    conf_scores = {}
    for metric in metrics:
        conf_scores[metric] = conf_interval(0.95, y_test.shape[0], metrics[metric])
    
    metric_dict = [metrics, conf_scores]
    output_metrics(metric_dict, model_name, f'-{vectortype}')
    
    bin_matrix(y_test, y_pred, model_type, vectortype)

    return y_test, y_pred, y_score

if __name__ ==  "__main__":
    hyperparams = {} # e.g. {'n_est': 50, 'learn_rate': 1} depending on model, see init_model function
    vectortype = 'countvec' # 'countvec' or 'tfidf'
    model_type = 'lr' # 'lr', 'svm', 'knn', 'rf', 'ada', or 'gb'
    bin_model(model_type, vectortype, hyperparams)
