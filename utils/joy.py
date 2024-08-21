import glob
import os
import time
from argparse import Namespace
from pathlib import Path

import torch
import torch.amp.autocast_mode
from PIL import Image
from torch import nn
from tqdm import tqdm
from transformers import (AutoModel, AutoProcessor, AutoTokenizer,
                          PreTrainedTokenizer, PreTrainedTokenizerFast, AutoModelForCausalLM)

from utils.image import image_process
from utils.logger import Logger

SUPPORT_IMAGE_FORMATS = ("bmp", "jpg", "jpeg", "png")

class ImageAdapter(nn.Module):
    def __init__(self, input_features: int, output_features: int):
        super().__init__()
        self.linear1 = nn.Linear(input_features, output_features)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(output_features, output_features)

    def forward(self, vision_outputs: torch.Tensor):
        x = self.linear1(vision_outputs)
        x = self.activation(x)
        x = self.linear2(x)
        return x

class Joy:
    def __init__(
            self,
            logger: Logger,
            args: Namespace,
            image_adapter_path: Path,
            clip_path: Path,
            llm_path: Path,
            use_gpu: bool = True,
    ):
        self.logger = logger
        self.args = args
        self.model_name = self.args.model_name
        self.image_adapter_path = image_adapter_path
        self.clip_path = clip_path
        self.llm_path = llm_path
        self.image_adapter = None
        self.clip_processor = None
        self.clip_model = None
        self.llm_tokenizer = None
        self.llm = None
        self.use_gpu = use_gpu

    def load_model(self):
        # Load CLIP
        self.logger.info(f'Loading CLIP with {"GPU" if self.use_gpu else "CPU"}...')
        start_time = time.monotonic()
        self.clip_processor = AutoProcessor.from_pretrained(self.clip_path)
        self.clip_model = AutoModel.from_pretrained(self.clip_path)
        self.clip_model = self.clip_model.vision_model
        self.clip_model.eval()
        self.clip_model.requires_grad_(False)
        self.clip_model.to("cuda")
        self.logger.info(f'CLIP Loaded in {time.monotonic() - start_time:.1f}s.')

        # Load LLM
        self.logger.info(f'Loading LLM with {"GPU" if self.use_gpu else "CPU"}...')
        start_time = time.monotonic()
        self.llm_tokenizer = AutoTokenizer.from_pretrained(self.llm_path, use_fast=False)
        assert (isinstance(self.llm_tokenizer, PreTrainedTokenizer) or
                isinstance(self.llm_tokenizer, PreTrainedTokenizerFast)), \
            f"Tokenizer is of type {type(self.llm_tokenizer)}"
        self.llm = AutoModelForCausalLM.from_pretrained(self.llm_path, device_map="auto", torch_dtype=torch.bfloat16)
        self.llm.eval()
        self.logger.info(f'LLM Loaded in {time.monotonic() - start_time:.1f}s.')

        # Load Image Adapter
        self.logger.info(f'Loading Image Adapter with {"GPU" if self.use_gpu else "CPU"}...')
        self.image_adapter = ImageAdapter(self.clip_model.config.hidden_size, self.llm.config.hidden_size)
        self.image_adapter.load_state_dict(torch.load(self.image_adapter_path, map_location="cpu"))
        self.image_adapter.eval()
        self.image_adapter.to("cuda")
        self.logger.info(f'Image Adapter Loaded in {time.monotonic() - start_time:.1f}s.')

    def inference(self):
        # Get image paths
        path_to_find = os.path.join(self.args.data_path, '**') \
            if self.args.recursive else os.path.join(self.args.data_path, '*')
        image_paths = sorted(set(
            [image for image in glob.glob(path_to_find, recursive=self.args.recursive)
             if image.lower().endswith(SUPPORT_IMAGE_FORMATS)]),
            key=lambda filename: (os.path.splitext(filename)[0])
        ) if not os.path.isfile(self.args.data_path) else str(self.args.data_path) \
            if str(self.args.data_path).lower().endswith(SUPPORT_IMAGE_FORMATS) else None

        if image_paths is None:
            self.logger.error('Invalid dir or image path!')
            raise FileNotFoundError

        self.logger.info(f'Found {len(image_paths)} image(s).')

        def get_caption(
                image: Image,
                user_prompt: str,
                temperature: float = 0.5,
                max_new_tokens: int = 300,
        ) -> str:
            # Cleaning VRAM cache
            torch.cuda.empty_cache()
            # Preprocess image
            image = self.clip_processor(images=image, return_tensors='pt').pixel_values
            image = image.to('cuda')
            # Tokenize the prompt
            prompt = self.llm_tokenizer.encode(user_prompt,
                                               return_tensors='pt',
                                               padding=False,
                                               truncation=False,
                                               add_special_tokens=False)
            # Embed image
            with torch.amp.autocast_mode.autocast('cuda', enabled=True):
                vision_outputs = self.clip_model(pixel_values=image, output_hidden_states=True)
                image_features = vision_outputs.hidden_states[-2]
                embedded_images = self.image_adapter(image_features)
                embedded_images = embedded_images.to('cuda')
            # Embed prompt
            prompt_embeds = self.llm.model.embed_tokens(prompt.to('cuda'))
            assert prompt_embeds.shape == (1, prompt.shape[1],
                                           self.llm.config.hidden_size), \
                f"Prompt shape is {prompt_embeds.shape}, expected {(1, prompt.shape[1], self.llm.config.hidden_size)}"
            embedded_bos = self.llm.model.embed_tokens(torch.tensor([[self.llm_tokenizer.bos_token_id]],
                                                                    device=self.llm.device,
                                                                    dtype=torch.int64))
            # Construct prompts
            inputs_embeds = torch.cat([
                embedded_bos.expand(embedded_images.shape[0], -1, -1),
                embedded_images.to(dtype=embedded_bos.dtype),
                prompt_embeds.expand(embedded_images.shape[0], -1, -1),
            ], dim=1)

            input_ids = torch.cat([
                torch.tensor([[self.llm_tokenizer.bos_token_id]], dtype=torch.long),
                torch.zeros((1, embedded_images.shape[1]), dtype=torch.long),
                prompt,
            ], dim=1).to('cuda')
            attention_mask = torch.ones_like(input_ids)
            # Generate caption
            generate_ids = self.llm.generate(input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                                             max_new_tokens=max_new_tokens, do_sample=True, top_k=10,
                                             temperature=temperature, suppress_tokens=None)
            # Trim off the prompt
            generate_ids = generate_ids[:, input_ids.shape[1]:]
            if generate_ids[0][-1] == self.llm_tokenizer.eos_token_id:
                generate_ids = generate_ids[:, :-1]

            content = self.llm_tokenizer.batch_decode(generate_ids,
                                                      skip_special_tokens=False,
                                                      clean_up_tokenization_spaces=False)[0]

            return content.strip()

        pbar = tqdm(total=len(image_paths), smoothing=0.0)
        for image_path in image_paths:
            try:
                pbar.set_description('Processing: {}'.format(image_path if len(image_path) <= 40 else
                                                             image_path[:15]) + ' ... ' + image_path[-20:])
                image = Image.open(image_path)
                image = image_process(image, int(self.args.image_size))
                caption = get_caption(
                    image=image,
                    user_prompt=self.args.user_prompt,
                    temperature=self.args.temperature,
                    max_new_tokens=self.args.max_tokens
                )
                if self.args.custom_caption_save_path is not None:
                    if not os.path.exists(self.args.custom_caption_save_path):
                        self.logger.error(f'{self.args.custom_caption_save_path} NOT FOUND!')
                        raise FileNotFoundError

                    self.logger.debug(f'Caption file(s) will be saved in {self.args.custom_caption_save_path}')

                    if os.path.isfile(self.args.data_path):
                        caption_file = str(os.path.splitext(os.path.basename(image_path))[0])

                    else:
                        caption_file = os.path.splitext(str(image_path)[len(str(self.args.data_path)):])[0]

                    caption_file = caption_file[1:] if caption_file[0] == '/' else caption_file
                    caption_file = os.path.join(self.args.custom_caption_save_path, caption_file)
                    # Make dir if not exist.
                    os.makedirs(os.path.join(str(caption_file)[:-len(os.path.basename(caption_file))]), exist_ok=True)
                    caption_file = Path(str(caption_file) + self.args.caption_extension)

                else:
                    caption_file = os.path.splitext(image_path)[0] + self.args.caption_extension

                if self.args.not_overwrite and os.path.isfile(caption_file):
                    self.logger.warning(f'Caption file {caption_file} already exist! Skip this caption.')
                    continue

                with open(caption_file, "wt", encoding="utf-8") as f:
                    f.write(caption + "\n")
                    self.logger.debug(f"\tImage path: {image_path}")
                    self.logger.debug(f"\tCaption path: {caption_file}")
                    self.logger.debug(f"\tCaption content: {caption}")

                pbar.update(1)

            except Exception as e:
                self.logger.error(f"Could not load image path: {image_path}, skip it.\nerror info: {e}")
                continue

        pbar.close()

    def unload_model(self):
        # Unload Image Adapter
        if self.image_adapter is not None:
            self.logger.info(f'Unloading Image Adapter...')
            start = time.monotonic()
            del self.image_adapter
            self.logger.info(f'Image Adapter unloaded in {time.monotonic() - start:.1f}s.')
        # Unload LLM
        if self.llm is not None:
            self.logger.info(f'Unloading LLM...')
            start = time.monotonic()
            del self.llm
            del self.llm_tokenizer
            self.logger.info(f'LLM unloaded in {time.monotonic() - start:.1f}s.')
        # Unload CLIP
        if self.clip_model is not None:
            self.logger.info(f'Unloading CLIP...')
            start = time.monotonic()
            del self.clip_model
            del self.clip_processor
            self.logger.info(f'CLIP unloaded in {time.monotonic() - start:.1f}s.')
