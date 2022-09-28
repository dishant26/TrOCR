import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import TrOCRProcessor
from transformers import VisionEncoderDecoderModel
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_metric
from transformers import default_data_collator

with open('/home/venkat/trocr/vocab.txt') as f:
    vocab = f.readlines()

with open('/home/venkat/trocr/train.txt') as f:
    train = f.readlines()

train_list = []
for i in range(len(train)):
    image_id = train[i].split(",")[0].split('/')[1].strip()
    vocab_id = int(train[i].split(",")[1].strip())
    text = vocab[vocab_id].split('\n')[0]
    row = [image_id, text]
    train_list.append(row)

train_df = pd.DataFrame(train_list, columns=['file_name', 'text'])

with open('/home/venkat/trocr/test.txt') as f:
    test = f.readlines()

test_list = []
for i in range(len(test)):
    image_id = test[i].split(",")[0].split('/')[1].strip()
    vocab_id = int(test[i].split(",")[1].strip())
    text = vocab[vocab_id].split('\n')[0]
    row = [image_id, text]
    test_list.append(row)

test_df = pd.DataFrame(test_list, columns=['file_name', 'text'])

with open('/home/venkat/trocr/val.txt') as f:
    val = f.readlines()

val_list = []
for i in range(len(val)):
    image_id = val[i].split(",")[0].split('/')[1].strip()
    vocab_id = int(val[i].split(",")[1].strip())
    text = vocab[vocab_id].split('\n')[0]
    row = [image_id, text]
    val_list.append(row)

val_df = pd.DataFrame(val_list, columns=['file_name', 'text'])

print(f"Train, Test, Val :{train_df.shape, test_df.shape, val_df.shape}")

class IAMDataset(Dataset):
    def __init__(self, root_dir, df, processor, max_target_length=128):
        self.root_dir = root_dir
        self.df = df
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # get file name + text 
        file_name = self.df['file_name'][idx]
        text = self.df['text'][idx]
        # prepare image (i.e. resize + normalize)
        image = Image.open(self.root_dir + file_name).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        # add labels (input_ids) by encoding the text
        labels = self.processor.tokenizer(text, 
                                          padding="max_length", 
                                          max_length=self.max_target_length).input_ids
        # important: make sure that PAD tokens are ignored by the loss function
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]

        encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}
        return encoding

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
train_dataset = IAMDataset(root_dir='/home/venkat/trocr/train/',
                           df=train_df,
                           processor=processor)
eval_dataset = IAMDataset(root_dir='/home/venkat/trocr/val/',
                           df=val_df,
                           processor=processor)

print("Number of training examples:", len(train_dataset))
print("Number of validation examples:", len(eval_dataset))

encoding = train_dataset[0]
for k,v in encoding.items():
  print(k, v.shape)

model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-stage1")

# set special tokens used for creating the decoder_input_ids from the labels
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
# make sure vocab size is set correctly
model.config.vocab_size = model.config.decoder.vocab_size

# set beam search parameters
model.config.eos_token_id = processor.tokenizer.sep_token_id
model.config.max_length = 64
model.config.early_stopping = True
model.config.no_repeat_ngram_size = 3
model.config.length_penalty = 2.0
model.config.num_beams = 4
os.environ["WANDB_DISABLED"] = "true"

training_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,
    evaluation_strategy="steps",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    output_dir="./",
    logging_steps=2,
    save_steps=1000,
    eval_steps=200,
)

cer_metric = load_metric("cer")

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    return {"cer": cer}

# instantiate trainer
trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=processor.feature_extractor,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=default_data_collator,
)
trainer.train()