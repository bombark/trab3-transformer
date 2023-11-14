import pandas as pd
import numpy as np
import torch

from datasets import load_dataset
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from torchinfo import summary


# cria o arquivo data.csv
# !wget -nc https://www.dropbox.com/s/lkd0eklmi64m9xm/AirlineTweets.csv?dl=0
#
# df = pd.read_csv('airlines.csv')
# target_map = { 'positive': 1, 'negative': 0, 'neutral': 2}
# df['target'] = df['airline_sentiment'].map(target_map)
# df1 = df[['text','target']]
# df1.columns = ['sentence','label']
# df1.to_csv('data.csv', index = False)


raw_dataset = load_dataset('csv', data_files = 'data.csv')
split = raw_dataset['train'].train_test_split(test_size=0.5, seed=42)


checkpoint = 'bert-base-cased'
tokernizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize_fn(batch):
	return tokernizer(batch['sentence'], truncation = True)
     
tokenized_dataset = split.map(tokenize_fn, batched = True)


model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels = 3)

summary(model)

training_args = TrainingArguments(
	output_dir='training_dir',
	evaluation_strategy='epoch',
	save_strategy='epoch',
	num_train_epochs=3,
	per_device_train_batch_size=16,
	per_device_eval_batch_size=64,
	)
     
def compute_metrics(logits_and_labels):
	logits, labels = logits_and_labels
	predictions = np.argmax(logits, axis=-1)
	acc = np.mean(predictions == labels)
	f1 = f1_score(labels, predictions, average = 'micro')
	return {'accuracy': acc, 'f1_score': f1}

trainer = Trainer(
	model,
	training_args,
	train_dataset = tokenized_dataset["train"],
	eval_dataset = tokenized_dataset["test"],
	tokenizer=tokernizer,
	compute_metrics=compute_metrics
	)
     
trainer.train()
