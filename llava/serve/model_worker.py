"""
A model worker executes the model.
"""
import time
from PIL import Image
import torch

from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token, KeywordsStoppingCriteria
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from transformers import TextIteratorStreamer


@torch.inference_mode()
def generate_stream(
        prompt, temperature, top_p, max_new_tokens, stop, tokenizer, model, image_processor, pil_images: [Image],
        is_multimodal=True, device='cuda'
):
    ori_prompt = prompt
    images = pil_images
    num_image_tokens = 0
    if images is not None and len(images) > 0 and is_multimodal:
        if len(images) > 0:
            if len(images) != prompt.count(DEFAULT_IMAGE_TOKEN):
                raise ValueError("Number of images does not match number of <image> tokens in prompt")
            images = process_images(images, image_processor, model.config)

            if type(images) is list:
                images = [image.to(model.device, dtype=torch.float16) for image in images]
            else:
                images = images.to(model.device, dtype=torch.float16)

            replace_token = DEFAULT_IMAGE_TOKEN
            if getattr(model.config, 'mm_use_im_start_end', False):
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

            num_image_tokens = prompt.count(replace_token) * model.get_vision_tower().num_patches
        else:
            images = None
    else:
        images = None


    temperature = float(temperature)
    top_p = float(top_p)
    max_context_length = getattr(model.config, 'max_position_embeddings', 2048)
    max_new_tokens = min(int(max_new_tokens), 1024)
    stop_str = stop
    do_sample = True if temperature > 0.001 else False

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device)
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=15)

    max_new_tokens = min(max_new_tokens, max_context_length - input_ids.shape[-1] - num_image_tokens)

    if max_new_tokens < 1:
        return  ori_prompt + "Exceeds max token length. Please start a new conversation, thanks."

    start1 = time.time()
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

    generated_text = ori_prompt
    for new_text in streamer:
        generated_text += new_text
        if generated_text.endswith(stop_str):
            generated_text = generated_text[:-len(stop_str)]

    return generated_text


if __name__ == "__main__":
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path='liuhaotian/llava-v1.5-13b',
        model_base=None,
        model_name='llava-v1.5-13b',
        load_8bit=False,
        load_4bit=False,
        device='cuda'
    )
    prompt = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\nwho is this? ASSISTANT:"
    temperature = 0.2
    top_p = 0.7
    max_new_tokens = 512
    stop = '</s>'
    pil_images = [Image.open("/opt/LLaVA/images/shaked.png")]

    start = time.time()
    x = generate_stream(prompt, temperature, top_p, max_new_tokens, stop, tokenizer, model, image_processor, pil_images)
    print(x)
    print(f"process took {time.time() - start}")
