from contextlib import contextmanager

import torch

from tether.baselines import authors

LAYER = 25
ALPHA = 5.0


class FixedBatch:
    def __init__(self, batch):
        self.batch = batch

    def __call__(self, **kwargs):
        return dict(self.batch)


@contextmanager
def steer(adapter, batch):
    m = authors.rudder()
    attention = m.get_qwenvl_self_attn(adapter.model, LAYER)
    with torch.no_grad():
        card = m.compute_card_vector_batch(
            adapter.model,
            FixedBatch(batch),
            images=[None],
            questions=["native prompt"],
            layer_idx=LAYER,
            pooling="attn",
            local=False,
        )
    m.mask_carrier.set(batch["attention_mask"].unsqueeze(-1).float())
    hook = m.BayesianGatingHookMaskedDynamic(
        max_alpha=ALPHA,
        sensitivity=5.0,
        concentration=1.0,
        carrier=m.mask_carrier,
        clamp=(0.05, 1.0),
        rms_match=False,
        record=False,
    )
    hook.set_vector(card)
    handle = attention.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()
        m.mask_carrier.clear()
