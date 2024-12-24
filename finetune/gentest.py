import torch
import textwrap
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

original_path = "/data0/lygao/model/llama/llama68m"
finetuned_path = "/data0/lygao/model/llama/llama68m-oasst"


model = AutoModelForCausalLM.from_pretrained(finetuned_path)
original_model = AutoModelForCausalLM.from_pretrained(original_path)
tokenizer = AutoTokenizer.from_pretrained(finetuned_path)

device = 4 if torch.cuda.is_available() else -1
text_generation = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)
original_text_generation = pipeline("text-generation", model=original_model, tokenizer=tokenizer, device=device)

prompt = "Calculate the total surface area of a cube with a side length of 5 cm."
generated_text = text_generation(prompt, max_new_tokens=64, temperature=2.0, top_p=0.9)[0]["generated_text"]
original_generated_text = original_text_generation(prompt, max_new_tokens=64, temperature=2.0, top_p=0.9)[0]["generated_text"]

# Format the output for better readability
wrapped_text = textwrap.fill(generated_text, width=100)
original_wrapped_text = textwrap.fill(original_generated_text, width=100)

print("\nGenerated Text:\n")
print(wrapped_text)
print("\n")
print(original_wrapped_text)
