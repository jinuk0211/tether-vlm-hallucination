import torch
import torch.nn.functional as F
from transformers import LogitsProcessorList

from tether.utils import gray_image

_linear = F.linear


def _rowwise_linear(x, weight, bias=None):
    if x.dtype == torch.bfloat16 and x.dim() == 3 and x.shape[0] == 2 and x.shape[1] == 1:
        return torch.cat([_linear(x[i : i + 1], weight, bias) for i in range(2)], 0)
    return _linear(x, weight, bias)


F.linear = _rowwise_linear


def stack(first, second):
    assert first.keys() == second.keys()
    assert torch.equal(first["input_ids"], second["input_ids"])
    return {k: torch.cat([first[k], second[k]], 0) for k in first if k != "labels"}


def gray_pair(adapter, ref):
    return stack(adapter.prepare(ref["image"], ref["prompt"]), adapter.prepare(gray_image(), ref["prompt"]))


@torch.no_grad()
def decode(adapter, batch, processor, max_new_tokens):
    n = batch["input_ids"].shape[1]
    out = adapter.model.generate(
        **batch,
        do_sample=False,
        num_beams=1,
        use_cache=True,
        max_new_tokens=max_new_tokens,
        logits_processor=LogitsProcessorList([processor]),
    )
    assert torch.equal(out[0], out[1])
    ids = out[0, n:].tolist()
    pad = adapter.tokenizer.pad_token_id
    if pad in ids:
        ids = ids[: ids.index(pad)]
    return adapter.tokenizer.decode(ids, skip_special_tokens=True), len(ids)
