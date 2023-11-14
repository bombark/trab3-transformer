from transformers import pipeline
from datasets import load_dataset


raw_dataset = load_dataset('csv', data_files = 'data.csv')
split = raw_dataset['train'].train_test_split(test_size=0.5, seed=42)

saved_model = pipeline('text-classification', model = 'training_dir/checkpoint-1374')

prediction = saved_model(split['test']['sentence'])
print(prediction[:10])
