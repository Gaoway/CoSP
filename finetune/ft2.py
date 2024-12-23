import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the model and tokenizer
model_id = "/data0/lygao/model/llama/llama68m"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Set the padding token if it's not already set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

from datasets import load_dataset, load_metric

datasetname  = 'swag'
print(f"Loading {datasetname} dataset...")

if datasetname == 'piqa':
    dataset = load_dataset('piqa', cache_dir='/data0/amax/cache/dataset')
    train = dataset["train"]

    prompts = train["goal"]
    sol1 = train["sol1"]
    sol2 = train["sol2"]
    labels = train["label"]
    responses = [sol1[i] if labels[i] == 0 else sol2[i] for i in range(len(labels))]
elif datasetname == 'swag':
    dataset = load_dataset('swag', cache_dir='/data0/amax/cache/dataset')
    train = dataset["train"]
    prompts = train["startphrase"]
    end0 = train["ending0"]
    end1 = train["ending1"]
    end2 = train["ending2"]
    end3 = train["ending3"]
    labels = train["label"]
    responses = [end0[i] if labels[i] == 0 else end1[i] if labels[i] == 1 else end2[i] if labels[i] == 2 else end3[i] for i in range(len(labels))]
    
max_length = 256
tokenized_inputs = tokenizer(prompts, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
tokenized_labels = tokenizer(responses, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")["input_ids"]
tokenized_labels[tokenized_labels == tokenizer.pad_token_id] = -100

from torch.utils.data import Dataset

# Create a custom dataset
class CustomDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.inputs["input_ids"])

    def __getitem__(self, idx):
        return {
            "input_ids": self.inputs["input_ids"][idx],
            "attention_mask": self.inputs["attention_mask"][idx],
            "labels": self.labels[idx]
        }
    
# Instantiate the dataset
dataset = CustomDataset(tokenized_inputs, tokenized_labels)

from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq

# Data collator to handle padding
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True
)

# Increase the number of training epochs if necessary
training_args = TrainingArguments(
    output_dir=f"/data0/amax/git/CoSP/finetune/{datasetname}",
    eval_strategy="no",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    num_train_epochs=3,  # Increase to 5 or more
    weight_decay=0.01
)


# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator
)

# Start training
try:
    trainer.train()
except ValueError as e:
    print("\nError during training:")
    print(e)

# Save the model and tokenizer
model.save_pretrained(f"/data0/amax/git/CoSP/finetune/{datasetname}/trained_model")
tokenizer.save_pretrained(f"/data0/amax/git/CoSP/finetune/{datasetname}/trained_model")

print("Model and tokenizer saved successfully!")


# Save the model

