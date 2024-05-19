import torch
from paddleocr import PaddleOCR
from transformers import pipeline


ocr = PaddleOCR(use_angle_cls=True, lang='en', use_space_char=True, use_gpu=False)

img = 'test06big.jpg'
result = ocr.ocr(img, cls=True)

ocr_str = ""
# for i in range(len(result)):
#     res = result[i]
#     for line in res:
#         print(line)



for i in range(len(result[0])):
    ocr_str = ocr_str + result[0][i][1][0] + "\n "
print(ocr_str)

pipe = pipeline("text-generation", model="HuggingFaceH4/zephyr-7b-alpha", torch_dtype=torch.bfloat16, device_map="auto")

messages = [
    {
        "role": "system",
        "content": "You are a text converter which receives raw boarding pass OCR information as a string and returns a structured output by organising the information in the string.",
    },
    {"role": "user", "content": f"Extract the relevant information related to goods boxes/packages from this OCR data: {ocr_str}. ometimes there might be more than one package details"},
]

prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

outputs = pipe(prompt, max_new_tokens=100, temperature=0.2, top_k=50, top_p=0.95)
print(outputs[0])