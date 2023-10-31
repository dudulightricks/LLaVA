"""
A model worker executes the model.
"""
import time
from PIL import Image
import torch

from llava.model import LlavaLlamaForCausalLM
from llava.mm_utils import process_images, tokenizer_image_token, StoppingCriteria
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from transformers import AutoTokenizer


from queue import Queue
from typing import Optional


class BaseStreamer:
    """
    Base class from which `.generate()` streamers should inherit.
    """

    def put(self, value):
        """Function that is called by `.generate()` to push new tokens"""
        raise NotImplementedError()

    def end(self):
        """Function that is called by `.generate()` to signal the end of generation"""
        raise NotImplementedError()


class TextStreamer(BaseStreamer):
    def __init__(self, tokenizer: "AutoTokenizer", skip_prompt: bool = False, **decode_kwargs):
        self.tokenizer = tokenizer
        self.skip_prompt = skip_prompt
        self.decode_kwargs = decode_kwargs

        # variables used in the streaming process
        self.token_cache = []
        self.print_len = 0
        self.next_tokens_are_prompt = True

    def put(self, value):
        """
        Receives tokens, decodes them, and prints them to stdout as soon as they form entire words.
        """
        if len(value.shape) > 1 and value.shape[0] > 1:
            # value = value[0]
            # self.put(value[0])
            # self.put(value[1])
            # return
            raise ValueError("TextStreamer only supports batch size 1")
        elif len(value.shape) > 1:
            value = value[0]

        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return

        # Add the new token to the cache and decodes the entire thing.
        self.token_cache.extend(value.tolist())
        text = self.tokenizer.decode(self.token_cache, **self.decode_kwargs)

        # After the symbol for a new line, we flush the cache.
        if text.endswith("\n"):
            printable_text = text[self.print_len :]
            self.token_cache = []
            self.print_len = 0
        # If the last token is a CJK character, we print the characters.
        elif len(text) > 0 and self._is_chinese_char(ord(text[-1])):
            printable_text = text[self.print_len :]
            self.print_len += len(printable_text)
        # Otherwise, prints until the last space char (simple heuristic to avoid printing incomplete words,
        # which may change with the subsequent token -- there are probably smarter ways to do this!)
        else:
            printable_text = text[self.print_len : text.rfind(" ") + 1]
            self.print_len += len(printable_text)

        self.on_finalized_text(printable_text)

    def end(self):
        """Flushes any remaining cache and prints a newline to stdout."""
        # Flush the cache, if it exists
        if len(self.token_cache) > 0:
            text = self.tokenizer.decode(self.token_cache, **self.decode_kwargs)
            printable_text = text[self.print_len :]
            self.token_cache = []
            self.print_len = 0
        else:
            printable_text = ""

        self.next_tokens_are_prompt = True
        self.on_finalized_text(printable_text, stream_end=True)

    def on_finalized_text(self, text: str, stream_end: bool = False):
        """Prints the new text to stdout. If the stream is ending, also prints a newline."""
        print(text, flush=True, end="" if not stream_end else None)

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if (
            (cp >= 0x4E00 and cp <= 0x9FFF)
            or (cp >= 0x3400 and cp <= 0x4DBF)  #
            or (cp >= 0x20000 and cp <= 0x2A6DF)  #
            or (cp >= 0x2A700 and cp <= 0x2B73F)  #
            or (cp >= 0x2B740 and cp <= 0x2B81F)  #
            or (cp >= 0x2B820 and cp <= 0x2CEAF)  #
            or (cp >= 0xF900 and cp <= 0xFAFF)
            or (cp >= 0x2F800 and cp <= 0x2FA1F)  #
        ):  #
            return True

        return False


class TextIteratorStreamer(TextStreamer):
    def __init__(
        self, tokenizer: "AutoTokenizer", skip_prompt: bool = False, timeout: Optional[float] = None, **decode_kwargs
    ):
        super().__init__(tokenizer, skip_prompt, **decode_kwargs)
        self.text_queue = Queue()
        self.stop_signal = None
        self.timeout = timeout

    def on_finalized_text(self, text: str, stream_end: bool = False):
        """Put the new text in the queue. If the stream is ending, also put a stop signal in the queue."""
        self.text_queue.put(text, timeout=self.timeout)
        if stream_end:
            self.text_queue.put(self.stop_signal, timeout=self.timeout)

    def __iter__(self):
        return self

    def __next__(self):
        value = self.text_queue.get(timeout=self.timeout)
        if value == self.stop_signal:
            raise StopIteration()
        else:
            return value


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



class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        self.max_keyword_len = 0
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if len(cur_keyword_ids) > 1 and cur_keyword_ids[0] == tokenizer.bos_token_id:
                cur_keyword_ids = cur_keyword_ids[1:]
            if len(cur_keyword_ids) > self.max_keyword_len:
                self.max_keyword_len = len(cur_keyword_ids)
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # output_ids = output_ids[:1, :]
        # if output_ids.shape[0] != 1:
        #     return self(output_ids[:1, :], scores, **kwargs) and self(output_ids[1:, :], scores, **kwargs)
        assert output_ids.shape[0] == 1, f"Only support batch size 1 (yet) {output_ids.shape}"  # TODO
        offset = min(output_ids.shape[1] - self.start_len, self.max_keyword_len)
        self.keyword_ids = [keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids]
        for keyword_id in self.keyword_ids:
            if (output_ids[0, -keyword_id.shape[0]:] == keyword_id).all():
                return True
        outputs = self.tokenizer.batch_decode(output_ids[:, -offset:], skip_special_tokens=True)[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True
        return False

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

    if images is None or len(images) == 0 or not is_multimodal:
        images = None
    else:
        if len(images) != len(prompts):
            raise ValueError("Number of images does not match number of prompts")
        ids, prom, stop_crs, max_num_image_tokens = [], [], [], 0
        for prompt, prompt_images in zip(prompts, images):
            input_ids, prompt_images, num_image_tokens = \
                handle_single_prompt(prompt, prompt_images, device, image_processor, model, tokenizer)
            ids.append(input_ids)
            prom.append(prompt_images)

            max_new_tokens = min(max_new_tokens, max_context_length - input_ids.shape[-1] - num_image_tokens)
            if max_new_tokens < 1:
                return prompt + "Exceeds max token length. Please start a new conversation, thanks."

            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
            stop_crs.append(stopping_criteria)

        input_ids = torch.stack(ids, dim=0).squeeze()
        images = torch.stack(prom, dim=0).squeeze()
        stopping_criteria = BatchKeywordsStoppingCriteria(stop_crs)

    temperature = float(temperature)
    top_p = float(top_p)

    do_sample = True if temperature > 0.001 else False
    streamer = BatchTextStreamer(tokenizer, batch_size=2, stop_str=stop_str, skip_prompt=True, skip_special_tokens=True, timeout=15)

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
