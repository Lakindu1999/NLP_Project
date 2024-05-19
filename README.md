# NLP_Project
Real-time Inventory Analysis through Advanced Text &amp; Image Recognition and Predictive Data Analytics

#Implementation of PaddleOCR and Zephyr-7B
Let’s install all pre-requiste libraries for Zephyr-7B.
!pip install git+https://github.com/huggingface/transformers.git
!pip install accelerate
Pre-requisite libraries for PaddleOCR
!git clone https://github.com/PaddlePaddle/PaddleOCR.git
!python3 -m pip install paddlepaddle-gpu
!pip install "paddleocr>=2.0.1"
Now let’s get all the imports
import torch
from paddleocr import PaddleOCR
from transformers import pipeline
Now let’s write the OCR through PaddleOCR first.
ocr = PaddleOCR(use_angle_cls=True, lang='en',use_space_char=True,show_log=False,enable_mkldnn=True)

img_path = 'imgtest.jpg'
result = ocr.ocr(img_path, cls=True)

ocr_string = ""  
#Extract the text from the OCR result and concatenate it to ocr_string
for i in range(len(result[0])):
    ocr_string = ocr_string + result[0][i][1][0] + " "
In this phase, we initialize PaddleOCR and assign it to the ocr variable, configuring it with various parameters. While most parameters, such as use_angle_cls and use_space_char, are standard, an additional parameter called enable_mkldnn is included. This parameter enhances performance with minimal overhead, effectively providing a free performance boost. For a detailed explanation of each parameter, refer to the PaddleOCR documentation.
Once PaddleOCR is set up, we pass the image path, stored in the img_path variable, to the OCR. The OCR processes the image, and we compile all the detected text into a single variable called ocr_string.

Now let’s move onto the LLM code.
pipe = pipeline("text-generation", model="HuggingFaceH4/zephyr-7b-alpha", torch_dtype=torch.bfloat16, device_map="auto")
We use the pipeline from the Transformers library to download and use zephyr-7b-alpha model from HuggingFace.
# Each message can have 1 of 3 roles: "system" (to provide initial instructions), "user", or "assistant". For inference, make sure "user" is the role in the final message.
messages = [
    {
        "role": "system",
        "content": "You are a text converter which receives raw boarding pass OCR information as a string and returns a structured output by organising the information in the string.",
    },
    {"role": "user", "content": f"Extract the relevant information related to goods boxes/packages from this OCR data: {ocr_str}"},
]# We use the tokenizer's chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating
prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
In this step, we create a prompt specifically tailored to our use case. For this demonstration, which focuses on boarding passes, the prompts are customized accordingly. However, these prompts can be easily adapted to fit a variety of other use cases. After crafting the prompt, we use the pipeline class, previously defined, to format our message into a structure that Zephyr-7B-alpha can effectively interpret. The formatted prompt is then ready for processing.
Here is how prompt looks after it has been formatted.
<|system|>
You are a text converter which receives raw boarding pass OCR information as a string and returns a structured output by organising the information in the string.</s>
<|user|>
Extract the relevant information related to goods boxes/packages from this OCR data: {ocr_str}</s>
<|assistant|>

The next stage involves sending the formatted prompt to the Zephyr-7B model for information extraction.
outputs = pipe(prompt, max_new_tokens=100, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
print(outputs[0]["generated_text"])
Key variables in this process include max_new_tokens, which dictates the number of new tokens the model can generate. A higher value for this variable allows for more extensive and creative outputs from the model. The variables temperature, top_k, and top_p manage the randomness of the model's responses.
•	temperature (float, optional, defaults to 1.0): This parameter influences the randomness of predictions by scaling the logits before applying softmax. A higher value leads to more varied completions, while a lower value results in more confident and conservative outputs from the model.
•	top_k (int, optional, defaults to 50): This parameter sets the number of highest probability vocabulary tokens to retain for top-k filtering. During sampling, it limits the pool of words to the top k most probable choices.
•	top_p (float, optional, defaults to 1.0): Also known as nucleus sampling, this parameter is an alternative to top-k sampling. Instead of selecting from the top k words, top-p sampling selects from the smallest set of words whose cumulative probability exceeds the specified probability p.
Once the model has processed the prompt, we then extract and print the JSON output. This output contains all the relevant information extracted from the image.

This is how the entire end-to-end function will look.
def Image_to_JSON(image_path):
    # Perform OCR on the image and extract the text content
    result = ocr.ocr(image_path, cls=True)

    ocr_string = ""  # Stores the OCR content extracted from the image in a string which can be fed into ChatGPT

    # Extract the text from the OCR result and concatenate it to ocr_string
    for i in range(len(result[0])):
        ocr_string = ocr_string + result[0][i][1][0] + " "

messages = [
    {
        "role": "system",
        "content": "You are a text converter which receives raw boarding pass OCR information as a string and returns a structured output by organising the information in the string.",
    },
    {"role": "user", "content": f"Extract the relevant information related to goods boxes/packages from this OCR data: {ocr_str}"},
]
# We use the tokenizer's chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=1000, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
    print(outputs[0]["generated_text"])
    return outputs[0]["generated_text"]
