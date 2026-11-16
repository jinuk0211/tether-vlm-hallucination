import hashlib
import importlib.metadata
from pathlib import Path
import platform
import random


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _versions():
    result = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("torch", "torchvision", "transformers", "numpy", "Pillow",
                 "tether"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "not-installed"
    return result


def _seed(config):
    import torch
    seed = int(config.get("generation", {}).get("seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _model_signature(config):
    return {key: value for key, value in config.items()
            if key not in ("device", "local_files_only")}


def _revision(adapter):
    return getattr(adapter.model.config, "_commit_hash", None)


def _provenance(adapter, config, manifests, records, parent=None):
    groups = {str(record["group_id"]) for record in records}
    sources = [{"path": str(Path(path).resolve()), "sha256": _sha(path)}
               for path in manifests]
    if parent:
        groups.update(parent.get("training_group_ids", []))
        sources = list(parent.get("manifests", [])) + sources
    return {
        "model_config": config["model"],
        "checkpoint_revision": _revision(adapter),
        "manifests": sources,
        "record_ids": [str(record["id"]) for record in records],
        "training_group_ids": sorted(groups),
        "versions": _versions(),
        "full_config": config,
    }


def _validate_artifact(artifact, model_config, records=None, adapter=None):
    provenance = artifact.get("_provenance")
    if not isinstance(provenance, dict) or "model_config" not in provenance:
        raise ValueError("Artifact has no model/data provenance.")
    if _model_signature(provenance["model_config"]) != _model_signature(model_config):
        raise ValueError("Artifact model config differs from the requested checkpoint/preprocessing.")
    if records:
        test_groups = {str(row["group_id"]) for row in records if row["split"] == "test"}
        overlap = test_groups.intersection(provenance.get("training_group_ids", []))
        if overlap:
            raise ValueError(f"Test image groups overlap artifact training/calibration: {sorted(overlap)[:5]}")
    if adapter is not None:
        expected = provenance.get("checkpoint_revision")
        if expected and _revision(adapter) != expected:
            raise ValueError("Resolved checkpoint revision differs from the artifact.")
