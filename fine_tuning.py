import numpy as np
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import Trainer, TrainingArguments

def load_data(filepath,tokenizer,blocksize=128):                        #Load textual dataset from .txt files for training
  dataset=TextDataset(
      tokenizer=tokenizer,
      file_path=filepath,
      block_size=blocksize
  )
  return dataset


def load_data_collator(tokenizer, mlm=False):
  datacollator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=mlm)
  return datacollator

def train(file_path, model_name, output_dir, overwrite_output_dir, per_device_train_batch_size, num_train_epochs, save_steps, checkpoint):
  tokenizer=GPT2Tokenizer.from_pretrained("gpt2")
  train_dataset=load_data(file_path,tokenizer)
  data_collator=load_data_collator(tokenizer)
  tokenizer.save_pretrained(output_dir)
  model=GPT2LMHeadModel.from_pretrained(model_name)
  model.save_pretrained(output_dir)

  training_args=TrainingArguments(output_dir=output_dir, overwrite_output_dir=overwrite_output_dir,
                                  per_device_train_batch_size=per_device_train_batch_size, num_train_epochs=num_train_epochs,learning_rate=5e-7,evaluation_strategy="no")
  trainer= Trainer(model=model, args=training_args, data_collator=data_collator, train_dataset=train_dataset,)
  trainer.train()
  trainer.save_model()

file_path="inputnew.txt"
model_name="gpt2"
output_dir="fine_tuning_weights"
overwrite_output_dir=False
per_device_train_batch_size=1                                           #To keep the training environment same as that of knowledge distillation
num_train_epochs=25
save_steps=300

train(file_path=file_path,
      model_name=model_name,
      output_dir=output_dir,
      overwrite_output_dir=overwrite_output_dir,
      per_device_train_batch_size=per_device_train_batch_size,
      num_train_epochs=num_train_epochs,
      save_steps=save_steps,
      checkpoint=True)
