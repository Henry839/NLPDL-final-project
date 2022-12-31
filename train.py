
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
#training_corpus = ( dataset["train"][i : i + 1000]["text"]  for i in range(0, len(dataset["train"]), 1000))
if "label" in dataset['train'].features:
    dataset = dataset.remove_columns(['label'])
print(dataset)
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'
model_checkpoint = "roberta-base"  # to change
epoch = 5
#old_tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
#tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, 52000)
#tokenizer.save_pretrained('tokenizer')
tokenizer = RobertaTokenizer.from_pretrained("tokenizer")

def tokenize_function(examples):
    result = tokenizer(examples["text"],max_length = 128,truncation = True)
    return result
batch_size = 32
dataset = dataset.map(tokenize_function)

from transformers import DataCollatorForLanguageModeling


data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)
config = RobertaConfig.from_pretrained(model_checkpoint)
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint,config=config)
model.resize_token_embeddings(len(tokenizer))
model.to(device)
training_args = TrainingArguments(
    output_dir=f"post_train_more_epoch",
    overwrite_output_dir=True,
    evaluation_strategy="epoch",
    num_train_epochs = epoch,
    learning_rate=2e-5,
    weight_decay=0.01,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    report_to="wandb"
)
trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
    )
trainer.train()
