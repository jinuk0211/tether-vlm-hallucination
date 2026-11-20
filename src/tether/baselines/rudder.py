from contextlib import contextmanager

import torch.nn.functional as F

from tether.baselines.common import output_tensor, with_output

LAYER = 25
ALPHA = 5.0
K = 5.0
C = 1.0
CLIP = (0.05, 1.0)


def gate(x, card):
    psi = F.cosine_similarity(x, card, dim=-1).unsqueeze(-1)
    plus = F.softplus(K * psi + C)
    minus = F.softplus(-K * psi + C)
    return (plus / (plus + minus)).clamp(*CLIP)


@contextmanager
def steer(adapter):
    layer = adapter.layers[LAYER]
    update = layer.post_attention_layernorm if adapter.model_type == "gemma4" else layer.self_attn
    state = {}

    def read(module, args, output):
        state["x"] = output[:, -1:].float()

    def add(module, args, output):
        u = output_tensor(output)
        if "card" not in state:
            state["card"] = F.normalize(u.float().sum(1, keepdim=True), dim=-1)
        card = state["card"]
        steered = u.clone()
        steered[:, -1:] = (u[:, -1:].float() + ALPHA * gate(state["x"], card) * card).to(u.dtype)
        return with_output(output, steered)

    handles = [
        layer.input_layernorm.register_forward_hook(read),
        update.register_forward_hook(add),
    ]
    try:
        yield
    finally:
        for h in handles:
            h.remove()
