from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

def main():

    model_path = "/home/ljy/PycharmProjects/foundation_model_for_moqiu/Mistral-7B-v0.1"
    model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=bnb_config, device_map="auto")
    # model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    messages = [
        {"role": "user",
         "content": "write a python function to generate a list of random 1000 numbers between 1 and 10000?"}
    ]

    encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
    device = "cuda:0"
    model_inputs = encodeds.to(device)

    generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
    decoded = tokenizer.batch_decode(generated_ids)
    print(decoded[0])

if __name__ == "__main__":
    main()
