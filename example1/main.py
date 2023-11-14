from transformers import BertTokenizer
import os
from tokenizers import BertWordPieceTokenizer

# create the ./bert-it-1/vocab.txt
vocab_size = 30_522
max_length = 512
special_tokens = [
  "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "<S>", "<T>"
]
files = ["train_mini.txt"]

tokenizer = BertWordPieceTokenizer()
tokenizer.train(files=files, vocab_size=vocab_size, special_tokens=special_tokens)
tokenizer.enable_truncation(max_length=max_length)

os.makedirs("./bert-it-mini", exist_ok=True)
tokenizer.save_model("./bert-it-mini")

# load the bert-it model
tokenizer = BertTokenizer.from_pretrained('./bert-it-mini/vocab.txt', local_files_only=True)
tokens = tokenizer("my dog is [MASK]", return_tensors="pt")
print(tokens)
