import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import LlamaTokenizer, LlamaForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, load_metric
import numpy as np

# 数据预处理
def preprocess_function(examples, tokenizer, max_length=128):
    inputs = []
    labels = []
    for goal, sol1, sol2, label in zip(examples['goal'], examples['sol1'], examples['sol2'], examples['label']):
        # 为每个选项创建输入序列
        input1 = goal + " " + sol1
        input2 = goal + " " + sol2
        inputs.append(input1)
        inputs.append(input2)
        # 标签需要转换为0或1
        labels.append(label - 1)  # 假设label为1或2，转换为0或1
    tokenized = tokenizer(inputs, padding='max_length', truncation=True, max_length=max_length)
    # 将标签整理为每对选项的标签
    processed_labels = []
    for i in range(0, len(labels), 2):
        processed_labels.append(labels[i//2])
    tokenized["labels"] = processed_labels
    return tokenized



# Load the model and tokenizer
model_id = "/data0/lygao/model/llama/llama68m"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
model.to(device)

# Set the padding token if it's not already set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset('piqa', cache_dir='/data0/amax/cache/dataset')
print(dataset['train'][0])




