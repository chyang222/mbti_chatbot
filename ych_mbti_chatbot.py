import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("taeminlee/kodialogpt2-base")
model = AutoModelForCausalLM.from_pretrained("taeminlee/kodialogpt2-base")

tokenizer = PreTrainedTokenizerFast.from_pretrained("taeminlee/kodialogpt2-base", bos_token='</s>', eos_token='</s>', unk_token='<unk>', pad_token='<pad>', mask_token='<mask>')


inputs = tokenizer("안녕!", return_tensors="pt")
outputs = model(**inputs)


##오케이 chatbot은 word2vec 모델같은 느낌이 아니니까 그럴빠엔 기본 예제의 모델로 먼저 pytorch 명령어들을 익히자.
## 위의 모델로서 바꾸자 
