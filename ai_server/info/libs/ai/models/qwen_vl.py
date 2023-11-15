# *_*coding:utf-8 *_*
# @Author : YueMengRui
import time
import torch
import base64
from PIL import Image
from io import BytesIO
from copy import deepcopy
from typing import Optional, Tuple, List, Dict
from transformers.generation import GenerationConfig
from transformers import AutoModelForCausalLM, AutoTokenizer


def pil_base64(image: Image):
    img_buffer = BytesIO()
    image.save(img_buffer, format='JPEG')
    byte_data = img_buffer.getvalue()
    base64_str = base64.b64encode(byte_data).decode('utf-8')
    return base64_str


def base64_pil(base64_str: str):
    image = base64.b64decode(base64_str.encode('utf-8'))
    image = BytesIO(image)
    image = Image.open(image)
    return image


def torch_gc(device):
    if torch.cuda.is_available():
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


class QwenVL:

    def __init__(self, model_name_or_path, model_name, device='cuda', logger=None, **kwargs):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = device
        self.logger = logger
        self.max_length = 8192
        self.ENDOFTEXT = "<|endoftext|>"
        self.IMSTART = "<|im_start|>"
        self.IMEND = "<|im_end|>"

        self._load_model(model_name_or_path, device)

        self.max_new_tokens = self.model.generation_config.max_new_tokens
        if self.logger:
            self.logger.info(str({'max_length': self.max_length, 'max_new_tokens': self.max_new_tokens}))

        # warmup
        self.lets_chat(prompt=[{'image': 'static/ocr.png'}, {'text': '图片中有什么？'}],
                       history=None,
                       stream=False
                       )

    def _load_model(self, model_name_or_path, device='cuda', **kwargs):

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                          device_map="auto",
                                                          trust_remote_code=True,
                                                          bf16=True
                                                          ).eval()
        self.model.generation_config = GenerationConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.device = self.model.device

    def make_context(self,
                     query: str,
                     history: List[Tuple[str, str]] = None,
                     system: str = "You are a helpful assistant.",
                     max_window_size: int = 6144,
                     chat_format: str = "chatml",
                     ):
        if history is None:
            history = []

        true_history = []

        if chat_format == "chatml":
            im_start, im_end = "<|im_start|>", "<|im_end|>"
            im_start_tokens = [self.tokenizer.im_start_id]
            im_end_tokens = [self.tokenizer.im_end_id]
            nl_tokens = self.tokenizer.encode("\n")

            def _tokenize_str(role, content):
                return f"{role}\n{content}", self.tokenizer.encode(
                    role, allowed_special=set(self.tokenizer.IMAGE_ST)
                ) + nl_tokens + self.tokenizer.encode(content, allowed_special=set(self.tokenizer.IMAGE_ST))

            system_text, system_tokens_part = _tokenize_str("system", system)
            system_tokens = im_start_tokens + system_tokens_part + im_end_tokens

            raw_text = ""
            context_tokens = []

            for turn_query, turn_response in reversed(history):
                query_text, query_tokens_part = _tokenize_str("user", turn_query)
                query_tokens = im_start_tokens + query_tokens_part + im_end_tokens
                if turn_response is not None:
                    response_text, response_tokens_part = _tokenize_str(
                        "assistant", turn_response
                    )
                    response_tokens = im_start_tokens + response_tokens_part + im_end_tokens

                    next_context_tokens = nl_tokens + query_tokens + nl_tokens + response_tokens
                    prev_chat = (
                        f"\n{im_start}{query_text}{im_end}\n{im_start}{response_text}{im_end}"
                    )
                else:
                    next_context_tokens = nl_tokens + query_tokens + nl_tokens
                    prev_chat = f"\n{im_start}{query_text}{im_end}\n"

                current_context_size = (
                        len(system_tokens) + len(next_context_tokens) + len(context_tokens)
                )
                if current_context_size < max_window_size:
                    context_tokens = next_context_tokens + context_tokens
                    raw_text = prev_chat + raw_text
                    true_history.insert(0, [turn_query, turn_response])
                else:
                    break

            context_tokens = system_tokens + context_tokens
            raw_text = f"{im_start}{system_text}{im_end}" + raw_text
            context_tokens += (
                    nl_tokens
                    + im_start_tokens
                    + _tokenize_str("user", query)[1]
                    + im_end_tokens
                    + nl_tokens
                    + im_start_tokens
                    + self.tokenizer.encode("assistant")
                    + nl_tokens
            )
            raw_text += f"\n{im_start}user\n{query}{im_end}\n{im_start}assistant\n"

        elif chat_format == "raw":
            raw_text = query
            context_tokens = self.tokenizer.encode(raw_text)
        else:
            raise NotImplementedError(f"Unknown chat format {chat_format!r}")

        return raw_text, context_tokens, true_history

    def token_counter(self, prompt):
        return len(self.tokenizer(prompt, return_tensors="pt").input_ids[0])

    def lets_chat(self, prompt, history, stream, generation_configs={}, **kwargs):
        """
        :param prompt: [
            {'image': 'image local path or url'},
            {'text': ''},
        ]
        :param history:
        :param stream:
        :param generation_configs:
        :param kwargs:
        :return:
        """
        if not isinstance(history, List):
            history = []

        query = self.tokenizer.from_list_format(prompt)

        raw_text, context_tokens, history = self.make_context(query=query, history=history)
        prompt_tokens = len(context_tokens)
        if self.logger:
            self.logger.info({'raw_text': raw_text, 'prompt_tokens': prompt_tokens})

        if stream:
            def stream_generator():
                start = time.time()
                image_flag = 0
                for response in self.model.chat_stream(tokenizer=self.tokenizer, query=query, history=history,
                                                       **kwargs):

                    response = response.replace(self.IMEND, '').replace(self.IMSTART, '').replace(
                        self.ENDOFTEXT, '')
                    generation_tokens = self.token_counter(response)
                    time_cost = time.time() - start
                    average_speed = f"{generation_tokens / time_cost:.3f} token/s"
                    torch_gc(self.device)
                    temp_history = deepcopy(history)
                    temp_history.append((query, response))
                    resp = {"model_name": self.model_name,
                            "type": "text",
                            "answer": "",
                            "history": temp_history,
                            "time_cost": {"generation": f"{time_cost:.3f}s"},
                            "usage": {"prompt_tokens": prompt_tokens, "generation_tokens": generation_tokens,
                                      "total_tokens": prompt_tokens + generation_tokens,
                                      "average_speed": average_speed}}

                    if image_flag == 1:
                        resp.update({'answer': '正在绘图中......'})
                        yield resp
                    else:
                        if self.tokenizer.ref_start_tag in response or self.tokenizer.box_start_tag in response:
                            image_flag = 1
                            resp.update({'answer': '正在绘图中......'})
                            yield resp
                        else:
                            resp.update({'answer': response, 'history': temp_history})
                            yield resp

                if image_flag == 1:
                    # image = self.tokenizer.draw_bbox_on_latest_picture(response, temp_history)
                    # image.save('xx.jpg')
                    image = Image.fromarray(
                        self.tokenizer.draw_bbox_on_latest_picture(response, temp_history).get_image())
                    resp.update({"type": "image", "image": pil_base64(image)})
                    yield resp

            return stream_generator()
        else:
            start = time.time()
            response, history = self.model.chat(tokenizer=self.tokenizer,
                                                query=query,
                                                history=history,
                                                **kwargs)
            generation_tokens = self.token_counter(response)
            time_cost = time.time() - start
            average_speed = f"{generation_tokens / time_cost:.3f} token/s"
            torch_gc(self.device)
            resp = {"model_name": self.model_name,
                    "type": "text",
                    "answer": "",
                    "history": history,
                    "time_cost": {"generation": f"{time_cost:.3f}s"},
                    "usage": {"prompt_tokens": prompt_tokens, "generation_tokens": generation_tokens,
                              "total_tokens": prompt_tokens + generation_tokens, "average_speed": average_speed}}

            if self.tokenizer.ref_start_tag in response or self.tokenizer.box_start_tag in response:
                # image = self.tokenizer.draw_bbox_on_latest_picture(response, history)
                # image.save('yy.jpg')
                image = Image.fromarray(
                    self.tokenizer.draw_bbox_on_latest_picture(response, history).get_image())
                resp.update({"type": "image", "image": pil_base64(image)})
            else:
                resp.update({"answer": response})

            return resp
