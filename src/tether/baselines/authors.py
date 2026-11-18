import ast
import importlib
import importlib.util
import sys
import types
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F

from tether.paths import UPSTREAM


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_function(path, name, namespace):
    tree = ast.parse(path.read_text(), filename=str(path))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


@lru_cache(maxsize=None)
def rudder():
    return load_file("rudder_methods", UPSTREAM / "rudder/Qwen/qwen_methods.py")


@lru_cache(maxsize=None)
def grains():
    root = UPSTREAM / "grains/src"
    package = types.ModuleType("grains_gradient")
    package.__path__ = [str(root / "attribution/gradient")]
    sys.modules[package.__name__] = package
    ig = load_file("grains_gradient.utils", root / "attribution/gradient/utils.py")

    class IntegratedGradients(ig.IntegratedGradients):
        def attribute(self, inputs, *args, **kwargs):
            kwargs.setdefault("internal_batch_size", inputs.shape[0])
            return super().attribute(inputs, *args, **kwargs)

    ig.IntegratedGradients = IntegratedGradients
    grad = load_file("grains_gradient.vlm_grad", root / "attribution/gradient/vlm_grad.py")
    steering = load_file("grains_steering", root / "utils/steering.py")
    namespace = {"torch": torch, "np": np, "log_softmax": F.log_softmax, "vlm_grad": grad}
    for name in ("get_all_layer_hidden_states", "get_log_prob", "get_top_token_indices"):
        load_function(root / "utils/model.py", name, namespace)
    ablation = load_function(root / "token_ablation_vlm.py", "run_token_ablation_contrastive", namespace)
    return types.SimpleNamespace(steering=steering, ablation=ablation)


@lru_cache(maxsize=None)
def vti():
    sys.path.insert(0, str(UPSTREAM / "vti"))
    return importlib.import_module("vti_utils.utils"), importlib.import_module("vti_utils.llm_layers")


@lru_cache(maxsize=None)
def revis():
    sys.path.insert(0, str(UPSTREAM / "revis"))
    sys.modules.setdefault("qwen_vl_utils", types.SimpleNamespace(process_vision_info=None))
    layers = importlib.import_module("steering_utils.llm_layers")
    vectors = importlib.import_module("steering_utils.vector_calculator")
    return layers, vectors
