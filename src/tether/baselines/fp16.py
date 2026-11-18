import importlib
from contextlib import contextmanager

import torch

MODULES = {
    "qwen2_5_vl": "transformers.models.qwen2_5_vl.modeling_qwen2_5_vl",
    "qwen3_vl": "transformers.models.qwen3_vl.modeling_qwen3_vl",
}


@contextmanager
def fp32_attention_scores(model_type):
    module = importlib.import_module(MODULES[model_type])
    original = module.eager_attention_forward

    def forward(attn, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        if query.dtype != torch.float16:
            return original(attn, query, key, value, attention_mask, scaling, dropout, **kwargs)
        keys = module.repeat_kv(key, attn.num_key_value_groups)
        values = module.repeat_kv(value, attn.num_key_value_groups)
        weights = torch.matmul(query.float(), keys.float().transpose(2, 3)) * scaling
        if attention_mask is not None:
            weights = weights + attention_mask.float()
        probs = torch.nn.functional.softmax(weights, dim=-1, dtype=torch.float32).to(query.dtype)
        probs = torch.nn.functional.dropout(probs, p=dropout, training=attn.training)
        return torch.matmul(probs, values).transpose(1, 2).contiguous(), probs

    module.eager_attention_forward = forward
    try:
        yield
    finally:
        module.eager_attention_forward = original
