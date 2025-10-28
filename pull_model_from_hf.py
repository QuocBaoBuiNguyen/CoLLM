import torch
from transformers import LlamaTokenizer, LlamaForCausalLM, BitsAndBytesConfig

## v2 models
# model_path = 'openlm-research/open_llama_7b_v2'

## v1 models
model_path = 'openlm-research/open_llama_3b'
# model_path = 'openlm-research/open_llama_7b'
# model_path = 'openlm-research/open_llama_13b'

bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

tokenizer = LlamaTokenizer.from_pretrained(model_path)
model = LlamaForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16
).cuda()

prompt = 'Q: What is the largest animal?\nA:'
input_ids = tokenizer(prompt, return_tensors="pt").input_ids

generation_output = model.generate(
    input_ids=input_ids.cuda(), max_new_tokens=32
)
print(tokenizer.decode(generation_output[0]))

pretrained_path = "/content/open_llama_3b"
model.save_pretrained(pretrained_path)
tokenizer.save_pretrained(pretrained_path)