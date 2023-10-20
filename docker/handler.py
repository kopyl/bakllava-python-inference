import runpod

from transformers import AutoConfig, AutoTokenizer, BitsAndBytesConfig
from llava.model.language_model.llava_mistral import LlavaMistralForCausalLM
from llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria

from PIL import Image
import requests
from io import BytesIO

from llava.conversation import conv_templates, SeparatorStyle
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

import torch


def disable_redundant_torch_init_to_accelerate_model_creation():
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


disable_redundant_torch_init_to_accelerate_model_creation()


def load_model():
    model_path = "SkunkworksAI/BakLLaVA-1"
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4'
    )
    cfg_pretrained = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = LlavaMistralForCausalLM.from_pretrained(
        model_path,
        low_cpu_mem_usage=True,
        load_in_4bit=True,
        config=cfg_pretrained,
        quantization_config=quantization_config,
        device_map="auto"
    )
    vision_tower = model.get_vision_tower()
    vision_tower.load_model()
    vision_tower.to(device='cuda', dtype=torch.float16)
    image_processor = vision_tower.image_processor
    return model, tokenizer, image_processor


def load_image(image_url, image_processor):
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content)).convert('RGB')
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()
    return image_tensor


def make_prompt(text, tokenizer):
    query = DEFAULT_IMAGE_TOKEN + '\n' + text

    conv = conv_templates["llava_v1"].copy()
    conv.append_message(conv.roles[0], query)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    return input_ids, stopping_criteria, stop_str


model, tokenizer, image_processor = load_model()


def generate_response(image_url, text, temperature=0.2, max_new_tokens=1024):
    image_tensor = load_image(image_url, image_processor)
    input_ids, stopping_criteria, stop_str = make_prompt(text, tokenizer)
    
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=input_ids,
            images=image_tensor,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria])
        
    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()
    return outputs


def handler(event):
    print(f"event: {event}")
    _input = event.get("input")
    if _input is None:
        return {
            "error": "INPUT_NOT_PROVIDED",
        }
    if (image_url := _input.get("image_url")) is None:
        return {
            "error": "MODEL_NAME_NOT_PROVIDED",
        }
    if (text := _input.get("text")) is None:
        return {
            "error": "TEXT_NOT_PROVIDED",
        }
    temperature = _input.get("temperature", 0.2)
    max_new_tokens = _input.get("max_new_tokens", 1024)
    response = generate_response(image_url, text, temperature, max_new_tokens)
    return response
    

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})


# !python runpod-handler.py --test_input '{"input": {"image_url": "https://t4.ftcdn.net/jpg/00/97/58/97/360_F_97589769_t45CqXyzjz0KXwoBZT9PRaWGHRk5hQqQ.jpg", "text": "Describe this image"}}'