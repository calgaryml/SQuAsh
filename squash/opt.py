import torch
from transformers import OPTForCausalLM, GPT2Tokenizer
device = "cuda" # the device to load the model onto

model = OPTForCausalLM.from_pretrained("facebook/opt-350m", torch_dtype=torch.float16)
tokenizer = GPT2Tokenizer.from_pretrained("facebook/opt-350m")

prompt = ("A chat between a curious human and the Statue of Liberty.\n\nHuman: What is your name?\nStatue: I am the "   
        "Statue of Liberty.\nHuman: Where do you live?\nStatue: New York City.\nHuman: How long have you lived "
        "there?")


model_inputs = tokenizer([prompt], return_tensors="pt").to(device)
model.to(device)

generated_ids = model.generate(**model_inputs, max_new_tokens=30, do_sample=False)
print(tokenizer.batch_decode(generated_ids)[0])