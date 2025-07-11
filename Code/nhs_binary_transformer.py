import pandas as pd
import numpy as np
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoConfig, EarlyStoppingCallback
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, precision_recall_curve, average_precision_score
from sklearn.model_selection import train_test_split
import random
import matplotlib.pyplot as plt
import scikitplot as skplt
from scipy.special import softmax, expit
import scipy.stats
import sys


df_train = pd.read_excel("final_trainset.xlsx")
df_train['binary2_label'].value_counts()
df_train['ABSTRACT'] = df_train['ABSTRACT'].apply(str.lower)
df_test = pd.read_excel("final_testset.xlsx")
df_test['ABSTRACT'] = df_test['ABSTRACT'].apply(str.lower)

model_names = {'PubMedBERT': 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract', 'DMIS BioBERT': 'dmis-lab/biobert-base-cased-v1.2', 'BlueBERT': 'bionlp/bluebert_pubmed_uncased_L-12_H-768_A-12', 'DistilBERT': 'distilbert-base-uncased', 'BERT': 'bert-base-uncased', 'Roberta': 'roberta-base'}
def compute_metrics(eval_pred):
    #taken from hf docs
    labels = eval_pred.label_ids
    preds = eval_pred.predictions.argmax(axis=1)

    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)

    logits = eval_pred.predictions
    y_score = expit(logits[:, 1])
    
    aps = average_precision_score(labels, y_score)

    metric_dict = {'TestAccuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1': f1, 'AvgPrecisionScore': aps}
    return metric_dict, labels, preds, y_score

def tokenize_dataset(tokenizer, abstracts, labels):
    encodings = tokenizer(list(abstracts), max_length=512, truncation=True, padding=True)
    encodings["label"] = list(labels)
    return Dataset.from_dict(encodings)

def output_metrics(metric_dict, model_name):
    columns = ['TestAccuracy', 'Precision', 'Recall', 'F1', 'AvgPrecisionScore']
    out_df = pd.DataFrame(metric_dict, columns=columns)
    out_df.to_csv(f'{model_name}-FinalBinary.csv', index=False)

def conf_interval(confidence, y_shape, metric):
    z_value = scipy.stats.norm.ppf((1 + confidence) / 2.0)
    ci_length = z_value * np.sqrt((metric * (1 - metric)) / y_shape)
    return ci_length

def bin_matrix(y_test, y_pred, model_type):
    fig = plt.figure(figsize=(15,6))
    ax1 = fig.add_subplot(121)
    skplt.metrics.plot_confusion_matrix(y_test, y_pred,
                                    title="Binary Confusion Matrix",
                                    ax=ax1)
    plt.savefig(model_type + '_binmat.png')

def main(model_name):
    X = df_train['ABSTRACT'].values
    y = df_train['binary2_label'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    id2label = {
        0: "the paper is not a primary experimental study in rare disease or the study is not directly investigating the natural history of a disease",
        1: "its primary contribution centers on observing the time course of a rare disease"
    }
    label2id = {v: k for k, v in id2label.items()}

    tokenizer = AutoTokenizer.from_pretrained(model_names[model_name])
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_ds = tokenize_dataset(tokenizer, X_train, y_train)
    eval_ds = tokenize_dataset(tokenizer, X_test, y_test)
    
    model = AutoModelForSequenceClassification.from_pretrained(model_names[model_name], num_labels=2, id2label=id2label, label2id=label2id)

    training_args = TrainingArguments(
        output_dir=f"{model_name}-bin",
        learning_rate=3e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=6,
        weight_decay=1,
        evaluation_strategy="epoch",
        logging_steps=1,
        save_strategy="epoch",
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        load_best_model_at_end=True,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        # compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(2)],
    )

    trainer.train()

    X_holdout = df_test['ABSTRACT'].values
    y_holdout = df_test['binary2_label'].values

    holdout_ds = tokenize_dataset(tokenizer, X_holdout, y_holdout)
    predictions = trainer.predict(holdout_ds)
    out_mets, y_test, y_pred, y_score = compute_metrics(predictions)

    precision, recall, _ = precision_recall_curve(y_test, y_score)
    plt.plot(recall, precision, color='blue',
             label=r'Precision-Recall (AUC = %0.2f)' % (out_mets['AvgPrecisionScore']), lw=1)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{model_name} PR Curve')
    plt.legend(loc="lower right")
    plt.savefig(f'{model_name}-PRCurve-FinalBin.png')

    conf_scores = {}
    for metric in out_mets:
        conf_scores[metric] = conf_interval(0.95, y_holdout.shape[0], out_mets[metric])
    metric_dict = [out_mets, conf_scores]
    output_metrics(metric_dict, model_name)
    bin_matrix(y_test, y_pred, model_name)

hf_model = sys.argv[1]
main(str(hf_model))