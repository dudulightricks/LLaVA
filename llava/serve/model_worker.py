"""
A model worker executes the model.
"""
import time
from PIL import Image
import torch
from typing import Optional

from llava.model import LlavaLlamaForCausalLM
from llava.mm_utils import process_images, tokenizer_image_token, KeywordsStoppingCriteria
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from transformers import AutoTokenizer, TextIteratorStreamer

SPACE_TOKEN = 29871

class BatchTextStreamer:
    def __init__(
        self, tokenizer: "AutoTokenizer", batch_size: int, stop_str: str, skip_prompt: bool = False, timeout: Optional[float] = None,
            **decode_kwargs
    ):
        self.streamers = []
        self.batch_size = batch_size
        self.stop_str = stop_str
        for _ in range(batch_size):
            self.streamers.append(TextIteratorStreamer(tokenizer, skip_prompt, timeout, **decode_kwargs))

    def put(self, value):
        if self.batch_size == 1:
            self.streamers[0].put(value)
            return

        for i, x in enumerate(value):
            if x.dim() == 0:
                x = x.unsqueeze(0)
            self.streamers[i].put(x)

    def end(self):
        for streamer in self.streamers:
            streamer.end()

    def get_result(self, index):
        result = ""
        for value in self.streamers[index]:
            result += value
            if result.endswith(self.stop_str):
                result = result[:-len(self.stop_str)]
        return result


class BatchKeywordsStoppingCriteria:
    def __init__(self, stopping_criterias: [KeywordsStoppingCriteria]):
        self.stopping_criterias = stopping_criterias

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        result = True
        for i in range(output_ids.shape[0]):
            result = result and self.stopping_criterias[i](output_ids[i:i + 1, :], scores, **kwargs)
        return result

def load_pretrained_model_LLaVA_15(model_path, device_map="auto", device="cuda"):
    kwargs = {"device_map": device_map, "torch_dtype": torch.float16}
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = LlavaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
    model.resize_token_embeddings(len(tokenizer))

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    vision_tower.to(device=device, dtype=torch.float16)
    image_processor = vision_tower.image_processor

    return tokenizer, model, image_processor

def pad_tensors_list_to_size(tensors_list, target_size):
    padded_tensors_list = []
    for tensor in tensors_list:
        padding_size = target_size - tensor.size(1)
        if padding_size > 0:
            padding = torch.full((1, padding_size), fill_value=SPACE_TOKEN, dtype=tensor.dtype, device=tensor.device)
            padded_tensor = torch.cat((tensor, padding), dim=1)
        else:
            padded_tensor = tensor
        padded_tensors_list.append(padded_tensor)
    return padded_tensors_list

@torch.inference_mode()
def generate_stream(
        prompts, temperature, top_p, max_new_tokens, stop, tokenizer, model, image_processor, pil_images: [Image],
        is_multimodal=True, device='cuda'
):
    if type(prompts) is not list:
        prompts = [prompts]
    ori_prompts = prompts
    images = pil_images
    num_image_tokens = 0
    stop_str = stop
    keywords = [stop_str]
    max_context_length = getattr(model.config, 'max_position_embeddings', 2048)
    max_new_tokens = min(int(max_new_tokens), 1024)
    max_input_ids_len = 0

    if images is None or len(images) == 0 or not is_multimodal:
        images = None
    else:
        if len(images) != len(prompts):
            raise ValueError("Number of images does not match number of prompts")
        ids, prom, stop_crs, max_num_image_tokens = [], [], [], 0
        for prompt, prompt_images in zip(prompts, images):
            input_ids, prompt_images, num_image_tokens = \
                handle_single_prompt(prompt, prompt_images, device, image_processor, model, tokenizer)
            max_input_ids_len = max(max_input_ids_len, input_ids.shape[-1])
            ids.append(input_ids)
            prom.append(prompt_images)

            max_new_tokens = min(max_new_tokens, max_context_length - input_ids.shape[-1] - num_image_tokens)
            if max_new_tokens < 1:
                return prompt + "Exceeds max token length. Please start a new conversation, thanks."

            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
            stop_crs.append(stopping_criteria)

        ids = pad_tensors_list_to_size(ids, max_input_ids_len)
        input_ids = torch.cat(ids, dim=0)
        images = torch.cat(prom, dim=0)
        stopping_criteria = BatchKeywordsStoppingCriteria(stop_crs)

    temperature = float(temperature)
    top_p = float(top_p)

    do_sample = True if temperature > 0.001 else False
    streamer = BatchTextStreamer(tokenizer, batch_size=len(prompts), stop_str=stop_str, skip_prompt=True, skip_special_tokens=True, timeout=15)

    start1 = time.time()

    print(f"input_ids.shape {input_ids.shape} {images.shape}")
    model.generate(
        inputs=input_ids,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        streamer=streamer,
        stopping_criteria=[stopping_criteria],
        use_cache=True,
        images=images,
    )
    print(f"model infer time: {time.time() - start1}")

    generated_texts = ori_prompts
    for i in range(len(generated_texts)):
        generated_texts[i] += streamer.get_result(i)

    return generated_texts

def handle_single_prompt(prompt, prompt_images, device, image_processor, model, tokenizer):
    if type(prompt_images) is not list:
        prompt_images = [prompt_images]
    if len(prompt_images) != prompt.count(DEFAULT_IMAGE_TOKEN):
        raise ValueError("Number of images does not match number of <image> tokens in prompt")
    prompt_images = process_images(prompt_images, image_processor, model.config)

    if type(prompt_images) is list:
        prompt_images = [image.to(model.device, dtype=torch.float16) for image in prompt_images]
    else:
        prompt_images = prompt_images.to(model.device, dtype=torch.float16)

    replace_token = DEFAULT_IMAGE_TOKEN
    if getattr(model.config, 'mm_use_im_start_end', False):
        replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
    prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

    num_image_tokens = prompt.count(replace_token) * model.get_vision_tower().num_patches
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device)
    return input_ids, prompt_images, num_image_tokens

if __name__ == "__main__":
    tokenizer, model, image_processor = load_pretrained_model_LLaVA_15(
        model_path='liuhaotian/llava-v1.5-13b', device='cuda'
    )
    prompt = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\nwho is this? ASSISTANT:"
    prompt2 = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\nwhat is this? ASSISTANT:"

    temperature = 0.2
    top_p = 0.7
    max_new_tokens = 512
    stop = '</s>'
    pil_images = [Image.open("/opt/LLaVA/images/shaked.png"), Image.open("/opt/LLaVA/images/llava_logo.png")]

    start = time.time()
    x = generate_stream([prompt, prompt2], temperature, top_p, max_new_tokens, stop, tokenizer, model, image_processor, pil_images)
    print(f"process took {time.time() - start}")
    print(x)
