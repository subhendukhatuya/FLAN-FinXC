'''
Input: Sentence  ---> Target: Tag Doc
'''
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, default_data_collator, get_linear_schedule_with_warmup
from peft import get_peft_config, get_peft_model, LoraConfig, get_peft_model_state_dict, PrefixTuningConfig, TaskType
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from datasets import Dataset
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

import warnings

warnings.filterwarnings("ignore")

from wordsegment import load, segment

load()

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

device = "cuda"
model_name_or_path = "google/flan-t5-xl"
tokenizer_name_or_path = "google/flant5-xl"

text_column = "sentence"
label_column = "text_label"
max_length = 128
batch_size = 8
print("all package imported")


class TrainDataset(Dataset):
    """Tourism Dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.

        """
        self.tacos_df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.sentences = self.tacos_df['input_text']
        # self.labels = self.tacos_df['Sentiment']
        self.text_labels = self.tacos_df['Tag_Doc']

    def __len__(self):
        return len(self.tacos_df)

    def __getitem__(self, idx):

        sentence = self.sentences[idx]
        # label = self.labels[idx]
        text_label = self.text_labels[idx]
        print('text label', text_label)

        sample = {'sentence': sentence, 'text_label': text_label}

        return sample


class TestDataset(Dataset):
    """Tourism Dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.

        """
        self.tacos_df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.sentences = self.tacos_df['input_text']
        # self.labels = self.tacos_df['Sentiment']
        self.text_labels = self.tacos_df['Tag_Doc']

    def __len__(self):
        return len(self.tacos_df)


xbrl_dataset_train = TrainDataset(csv_file='./data/consolidated_xbrl_train.csv',
                                     root_dir='./')

xbrl_dataset_test = TestDataset(csv_file='./data/consolidated_xbrl_test.csv', root_dir='./')

dataset_train = Dataset.from_dict(
    {"sentence": list(xbrl_dataset_train.sentences), "text_label": list(xbrl_dataset_train.text_labels)})
dataset_test = Dataset.from_dict(
    {"sentence": list(xbrl_dataset_test.sentences), "text_label": list(xbrl_dataset_test.text_labels)})

print(dataset_train["text_label"])
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir='/')


def preprocess_function(examples):
    inputs = examples[text_column]
    targets = examples[label_column]
    model_inputs = tokenizer(inputs, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
    labels = tokenizer(targets, max_length=30, padding="max_length", truncation=True, return_tensors="pt")
    labels = labels["input_ids"]
    labels[labels == tokenizer.pad_token_id] = -100
    model_inputs["labels"] = labels
    return model_inputs


processed_datasets_train = dataset_train.map(
    preprocess_function,
    batched=True,
    num_proc=1,
    remove_columns=['sentence', 'text_label'],
    load_from_cache_file=False,
    desc="Running tokenizer on dataset",
)

processed_datasets_test = dataset_test.map(
    preprocess_function,
    batched=True,
    num_proc=1,
    remove_columns=['sentence', 'text_label'],
    load_from_cache_file=False,
    desc="Running tokenizer on dataset",
)

train_dataset = processed_datasets_train
eval_dataset = processed_datasets_test

train_dataloader = DataLoader(
    train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True
)

eval_dataloader = DataLoader(eval_dataset, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True)


lora_config = LoraConfig(r=2, lora_alpha=32, target_modules=["q", "v"], lora_dropout=0.05, bias="none",
                         task_type=TaskType.SEQ_2_SEQ_LM)

model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, cache_dir='/NS/ssdecl/work', device_map='auto')
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
"trainable params: 983040 || all params: 738651136 || trainable%: 0.13308583065659835"



output_dir = "lora-flan-t5-xl"

# Define training args
training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    auto_find_batch_size=True,
    learning_rate=5e-4,  # higher learning rate
    num_train_epochs=5,
    logging_dir=f"{output_dir}/logs",
    logging_strategy="steps",
    logging_steps=500,
    save_strategy="no",
)

# Create Trainer instance
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

# train model
trainer.train()
peft_model_id = "lora_flan_xl_model_sent_tag_doc"
trainer.model.save_pretrained(peft_model_id)
tokenizer.save_pretrained(peft_model_id)

model.eval()

eval_preds = []

for step, batch in enumerate(tqdm(eval_dataloader)):
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    eval_preds.extend \
        (tokenizer.batch_decode(torch.argmax(outputs.logits, -1).detach().cpu().numpy(), skip_special_tokens=True))

correct = 0
total = 0
f1 = open('./lora_prediction.txt', 'w')
for pred, true in zip(eval_preds, dataset_test["text_label"]):
    print(pred, true)
    f1.write('True: ' + true + ' Pred: ' + pred)
    f1.write('\n')
    if pred.strip() == true.strip():
        correct += 1
    total += 1
f1.close()
accuracy = correct / total * 100
print(f"{accuracy=} % on the evaluation dataset")

print("---Completed The Task -----")
