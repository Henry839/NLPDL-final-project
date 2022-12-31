from data import get_dataset
from model import get_model
from transformers import AutoTokenizer,RobertaTokenizer
from transformers import DataCollatorForLanguageModeling,DataCollatorWithPadding
from transformers import TrainingArguments
from transformers import RobertaConfig,AutoModelForMaskedLM
import evaluate
from transformers import Trainer

dataset_name = ["ai_corpus","citation_intent","sciie","computer science 0","computer science 1","computer science 3","computer science 4","computer science 5","computer science 6","computer science 7"]
dataset = get_dataset(dataset_name)
training_corpus = ( dataset["train"][i : i + 1000]["text"]  for i in range(0, len(dataset["train"]), 1000))
if "label" in dataset['train'].features:
    dataset = dataset.remove_columns(['label'])
print(dataset)
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'
model_checkpoint = "roberta-base"
epoch = 2
old_tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, 52000)
tokenizer.save_pretrained('tokenizer')
tokenizer = RobertaTokenizer.from_pretrained("tokenizer")