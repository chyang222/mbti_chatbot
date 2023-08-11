import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("taeminlee/kodialogpt2-base")
model = AutoModelForCausalLM.from_pretrained("taeminlee/kodialogpt2-base")


inputs = tokenizer("안녕!", return_tensors="pt")
outputs = model(**inputs)


predicted_token_indices = torch.argmax(outputs.logits, dim=-1)
predicted_texts = [tokenizer.decode(indices, skip_special_tokens=True) for indices in predicted_token_indices]

for text in predicted_texts:
    print(text)

with torch.no_grad():
    while 1:
        q = input("user > ").strip()
        if q == "quit":
            break
        a = ""
        while 1:
            input_ids = torch.LongTensor(model.encode(Q_TKN + q + SENT + sent + A_TKN + a)).unsqueeze(dim=0)
            pred = model(input_ids)
            pred = pred.logits
            gen = model.convert_ids_to_tokens(torch.argmax(pred, dim=-1).squeeze().numpy().tolist())[-1]
            if gen == EOS:
                break
            a += gen.replace("▁", " ")
        print("Chatbot > {}".format(a.strip()))