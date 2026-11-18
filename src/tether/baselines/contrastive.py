import argparse
import hashlib
import json
import math
import time

import torch
from transformers import LogitsProcessor

from tether import lockstep
from tether.models import MODELS, load
from tether.utils import add_run_args, run_rows, seed_everything


def add_diffusion_noise(image, step, generator=None, shape=None):
    betas = torch.sigmoid(torch.linspace(-6, 6, 1000)) * (0.5e-2 - 1e-5) + 1e-5
    alphas_prod = torch.cumprod(1 - betas, dim=0)
    noise = torch.randn(shape or image.shape, generator=generator, dtype=torch.float32)
    return torch.sqrt(alphas_prod)[step] * image.clone() + torch.sqrt(1 - alphas_prod)[step] * noise


def noise_seed(ref):
    return int(hashlib.sha256(("vcd:" + ref["id"]).encode()).hexdigest()[:8], 16)


def noised(adapter, pixels, ref, step):
    g = torch.Generator().manual_seed(noise_seed(ref))
    x = pixels.detach().float().cpu()
    if adapter.model_type == "gemma4":
        y = (add_diffusion_noise(2 * (x - 0.5), step, g) + 1) / 2
    else:
        ip = adapter.processor.image_processor
        T, P = int(ip.temporal_patch_size), int(ip.patch_size)
        v = x.view(x.shape[0], 3, T, P, P)
        y = add_diffusion_noise(v, step, g, shape=(x.shape[0], 3, 1, P, P)).reshape(x.shape)
    return y.to(pixels.device, pixels.dtype)


def image_span(full, text):
    n, m = len(full), len(text)
    a = 0
    while a < m and full[a] == text[a]:
        a += 1
    b = 0
    while b < m - a and full[n - 1 - b] == text[m - 1 - b]:
        b += 1
    assert a + b == m
    return a, n - b


def contrast_pair(adapter, ref, method, noise_step):
    real = adapter.prepare(ref["image"], ref["prompt"])
    other = dict(real)
    if method == "vcd":
        other["pixel_values"] = noised(adapter, real["pixel_values"], ref, noise_step)
    else:
        full = real["input_ids"][0].tolist()
        text = adapter.prepare(None, ref["prompt"])["input_ids"][0].tolist()
        a, b = image_span(full, text)
        mask = real["attention_mask"].clone()
        mask[0, a:b] = 0
        other["attention_mask"] = mask
    return lockstep.stack(real, other)


class VCD(LogitsProcessor):
    def __init__(self, alpha=1.0, beta=0.1):
        self.alpha = alpha
        self.beta = beta

    def __call__(self, input_ids, scores):
        real, noisy = scores[0].float(), scores[1].float()
        cutoff = math.log(self.beta) + real.max()
        cd = ((1 + self.alpha) * real - self.alpha * noisy).masked_fill(real < cutoff, -float("inf"))
        scores[0] = cd.to(scores.dtype)
        scores[1] = scores[0]
        return scores


class M3ID(LogitsProcessor):
    def __init__(self, prompt_len, lam=0.02, threshold=0.3):
        self.prompt_len = prompt_len
        self.lam = lam
        self.threshold = threshold

    def __call__(self, input_ids, scores):
        t = input_ids.shape[1] - self.prompt_len + 1
        with_image = scores[0].float().log_softmax(-1)
        without = scores[1].float().log_softmax(-1)
        out = with_image
        if with_image.max() < math.log(self.threshold):
            gamma = math.exp(-self.lam * t)
            out = with_image + (1 - gamma) / gamma * (with_image - without)
        scores[0] = out.to(scores.dtype)
        scores[1] = scores[0]
        return scores


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=MODELS, required=True)
    p.add_argument("--method", choices=["vcd", "m3id"], required=True)
    add_run_args(p)
    p.add_argument("--vcd-alpha", type=float, default=1.0)
    p.add_argument("--vcd-beta", type=float, default=0.1)
    p.add_argument("--vcd-noise", type=int, default=500)
    p.add_argument("--m3id-lam", type=float, default=0.02)
    p.add_argument("--m3id-threshold", type=float, default=0.3)
    p.add_argument("--attn")
    args = p.parse_args()

    refs, path = run_rows(args.data, args.output, args.method, args.start, args.end)
    adapter = load(args.model, attn=args.attn)
    with open(path, "a") as f:
        for ref in refs:
            seed_everything(42)
            start = time.perf_counter()
            batch = contrast_pair(adapter, ref, args.method, args.vcd_noise)
            if args.method == "vcd":
                processor = VCD(args.vcd_alpha, args.vcd_beta)
            else:
                processor = M3ID(batch["input_ids"].shape[1], args.m3id_lam, args.m3id_threshold)
            text, n = lockstep.decode(adapter, batch, processor, ref["max_new_tokens"])
            row = {
                **ref,
                "method": args.method,
                "prediction": text,
                "generated_tokens": n,
                "elapsed_seconds": time.perf_counter() - start,
            }
            f.write(json.dumps(row) + "\n")
            f.flush()


if __name__ == "__main__":
    main()
