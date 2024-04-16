from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import Trainer, TrainingArguments,DataCollatorForLanguageModeling
import torch
tokenizer=AutoTokenizer.from_pretrained("gpt2",padding_size="right",device_map="cuda",)
tokenizer.pad_token=tokenizer.eos_token
model=AutoModelForCausalLM.from_pretrained("gpt2",device_map="cuda",)

from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
model.gradient_checkpointing_enable()
model=prepare_model_for_kbit_training(model)

peft_config= LoraConfig(inference_mode=False, lora_alpha=32, lora_dropout=0.1, peft_type=TaskType.CAUSAL_LM)
model=get_peft_model(model,peft_config)
print(model.print_trainable_parameters())
from transformers import TextDataset
def load_data(filepath,tokenizer,blocksize=128):
  dataset=TextDataset(
      tokenizer=tokenizer,
      file_path=filepath,
      block_size=blocksize
  )
  return dataset
dataset=load_data("inputnew.txt",tokenizer)

datacollator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
output_dir="lora_weights"
overwrite_output_dir=False
per_device_train_batch_size=1                                              #To keep the training environment same for comparisopn
num_train_epochs=25                                                        #To keep training environment same for comparison
training_args=TrainingArguments(output_dir=output_dir,
                                overwrite_output_dir=overwrite_output_dir,
                                gradient_checkpointing=True,
                                gradient_accumulation_steps= 4,
                                learning_rate=5e-7,
                                logging_steps=5,
                                save_strategy="steps",
                                save_steps=50,
                                per_device_train_batch_size=per_device_train_batch_size, 
                                num_train_epochs=num_train_epochs,)
trainer= Trainer(model=model, args=training_args, data_collator=datacollator, train_dataset=dataset,)
trainer.train()
trainer.save_model()
#to merge both lora and original model
model.save_pretrained("loraupdated", safe_serialization=False, )
from peft import PeftModel
model = PeftModel.from_pretrained(model, "loraupdated",device_map="cuda")
model_ = model.merge_and_unload()                                          #Combines the Lora Weights with the original weights
model_.save_pretrained("merged_model_")

