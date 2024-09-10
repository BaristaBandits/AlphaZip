from transformers import TrainingArguments
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer
from transformers import TextDataset, DataCollatorForLanguageModeling


class KnowledgeDistillationTrainingArguments(
    TrainingArguments
):  # Defining the parameters for training
    def __init__(self, *args, alpha=0.5, temperature=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.temperature = temperature


class KnowledgeDistillationTrainer(Trainer):

    def __init__(self, *args, teacher_model="gpt2-xl", **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs_student = model(**inputs)
        student_loss = outputs_student.loss
        student_logits = outputs_student.logits
        teacher_logits = self.teacher_model(**inputs).logits

        distillation_loss = nn.KLDivLoss(reduction="batchmean")
        # Utilising reverse KL Divergence Loss
        loss_kd = self.args.temperature**2 * distillation_loss(
            F.softmax(student_logits / self.args.temperature, dim=1),
            F.softmax(teacher_logits / self.args.temperature, dim=1),
        )
        loss = self.args.alpha * student_loss + (1 - self.args.alpha) * loss_kd

        return loss


def load_data(
    filepath, tokenizer, blocksize=128
):  # Loads textual dataset for training from a file in .txt extension
    dataset = TextDataset(tokenizer=tokenizer, file_path=filepath, block_size=blocksize)
    return dataset


import numpy as np
from datasets import load_metric

batch_size = 1  # Works the best for NVIDIA RTX 4080 GPU
output_directory = "Knowledgedistillation_weights"

student_training_args = KnowledgeDistillationTrainingArguments(
    output_dir=output_directory,
    num_train_epochs=25,
    learning_rate=5e-7,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    alpha=1,
    evaluation_strategy="no",  # Optional: Can give a dataset for evaluation in eval_dataset in Trainer, set eval_strategy="epoch" in that case
    save_steps=1000,
    weight_decay=0.01,
)

from transformers import AutoModelForCausalLM, AutoTokenizer

file_path = "input.txt"
tokenizer = AutoTokenizer.from_pretrained("gpt2")
datacollator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
dataset = load_data(file_path, tokenizer)


def student_init():
    return AutoModelForCausalLM.from_pretrained("gpt2").to(device="cuda")


teacher_model = AutoModelForCausalLM.from_pretrained("gpt2-xl").to(
    device="cuda"
)  # Necessary to push models to gpu environment for faster training


gpt2_trainer = KnowledgeDistillationTrainer(
    model_init=student_init,
    teacher_model=teacher_model,
    args=student_training_args,
    train_dataset=dataset,
    data_collator=datacollator,
    tokenizer=tokenizer,
)

gpt2_trainer.train()
