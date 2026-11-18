from contextlib import ExitStack

import torch


def output_tensor(output):
    return output[0] if isinstance(output, (tuple, list)) else output


def with_output(output, value):
    if isinstance(output, tuple):
        return (value, *output[1:])
    if isinstance(output, list):
        return [value, *output[1:]]
    return value


@torch.no_grad()
def capture(adapter, batch, layers, vision=False, positions=False):
    modules = adapter.vision_layers if vision else adapter.layers
    states = {}
    with ExitStack() as stack:
        for i in layers:

            def hook(module, args, output, i=i):
                h = output_tensor(output)
                if vision:
                    h = h[0] if h.ndim == 3 else h
                    value = h if positions else h.mean(0)
                else:
                    value = h[0, -1]
                states[i] = value.detach().float().cpu().clone()

            stack.callback(modules[i].register_forward_hook(hook).remove)
        adapter.forward(batch)
    return states


def steer_last_token(module, update):
    def hook(m, args, output):
        h = output_tensor(output)
        changed = h.clone()
        changed[:, -1:, :] = update(h[:, -1:, :])
        return with_output(output, changed)

    return module.register_forward_hook(hook)
