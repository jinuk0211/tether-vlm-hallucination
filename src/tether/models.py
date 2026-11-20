import torch
from PIL import Image, ImageOps

CONFIGS = {
    "qwen": {
        "family": "qwen2_5_vl",
        "checkpoint": "Qwen/Qwen2.5-VL-7B-Instruct",
        "revision": "cc594898137f460bfe9f0759e9844b3ce807cfb5",
        "attn": "sdpa",
    },
    "qwen3": {
        "family": "qwen3_vl",
        "checkpoint": "Qwen/Qwen3-VL-8B-Instruct",
        "revision": "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
        "attn": "sdpa",
    },
    "gemma": {
        "family": "gemma4",
        "checkpoint": "google/gemma-4-E4B-it",
        "revision": "ee0ef6023621cff504d758262d4e04895a5af4a2",
        "attn": "eager",
    },
}
MODELS = tuple(CONFIGS)
DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


class Adapter:
    def __init__(self, model, processor, image_size=448):
        self.model = model.eval()
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.image_size = image_size
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        self.model_type = model.config.model_type
        lm = model.model.language_model
        self.layers = lm.layers
        self.norm = lm.norm
        if self.model_type == "gemma4":
            self.vision_layers = model.model.vision_tower.encoder.layers
        else:
            self.vision_layers = model.model.visual.blocks
        text_config = getattr(model.config, "text_config", model.config)
        self.hidden_size = text_config.hidden_size
        self.image_token_id = getattr(model.config, "image_token_id", None)
        if self.image_token_id is None:
            self.image_token_id = model.config.image_token_index
        self.yes_id = self.token_id("Yes")
        self.no_id = self.token_id("No")

    def token_id(self, text):
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        assert len(ids) == 1, (text, ids)
        return ids[0]

    def load_image(self, path):
        with Image.open(path) as f:
            image = ImageOps.exif_transpose(f).convert("RGB")
        return image.resize((self.image_size, self.image_size), Image.Resampling.BICUBIC)

    def chat(self, prompt, image=True):
        content = [{"type": "image"}] if image else []
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        return self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def prepare(self, image, prompt, answer=None):
        text = self.chat(prompt, image is not None)
        kwargs = {"return_tensors": "pt"}
        if image is not None:
            kwargs["images"] = [self.load_image(image)]
        batch = dict(self.processor(text=[text], **kwargs))
        if answer is not None:
            eos = self.tokenizer.eos_token or ""
            prompt_ids = batch["input_ids"]
            batch = dict(self.processor(text=[text + answer + eos], **kwargs))
            n = prompt_ids.shape[1]
            assert torch.equal(batch["input_ids"][:, :n], prompt_ids)
            labels = batch["input_ids"].clone()
            labels[:, :n] = -100
            batch["labels"] = labels
        return {
            k: v.to(self.device, self.dtype if v.is_floating_point() else v.dtype)
            for k, v in batch.items()
            if torch.is_tensor(v)
        }

    def forward(self, batch):
        return self.model(**batch, use_cache=False, return_dict=True)

    @torch.no_grad()
    def generate(self, batch, max_new_tokens):
        inputs = {k: v for k, v in batch.items() if k != "labels"}
        out = self.model.generate(
            **inputs, do_sample=False, num_beams=1, max_new_tokens=max_new_tokens, use_cache=True
        )
        ids = out[:, inputs["input_ids"].shape[1] :]
        text = self.processor.batch_decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[
            0
        ]
        return text, ids.shape[1]


def load(name, dtype="bfloat16", attn=None, device="cuda", device_map=None):
    from transformers import (
        AutoProcessor,
        Gemma4ForConditionalGeneration,
        Qwen2_5_VLForConditionalGeneration,
        Qwen3VLForConditionalGeneration,
    )

    classes = {
        "qwen2_5_vl": Qwen2_5_VLForConditionalGeneration,
        "qwen3_vl": Qwen3VLForConditionalGeneration,
        "gemma4": Gemma4ForConditionalGeneration,
    }
    config = CONFIGS[name]
    kwargs = {"revision": config["revision"]}
    processor = AutoProcessor.from_pretrained(config["checkpoint"], **kwargs)
    model = classes[config["family"]].from_pretrained(
        config["checkpoint"],
        dtype=DTYPES[dtype],
        attn_implementation=attn or config["attn"],
        device_map=device_map,
        **kwargs,
    )
    if device_map is None:
        model = model.to(device)
    return Adapter(model, processor)
