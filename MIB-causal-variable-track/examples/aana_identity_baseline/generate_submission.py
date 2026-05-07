#!/usr/bin/env python3
"""Generate a minimal AANA-style identity baseline submission for MIB.

The generated artifact is intentionally small and deterministic. It validates
the causal-variable-track submission interface without claiming leaderboard
performance.
"""

import argparse
import json
import shutil
from pathlib import Path

import torch


TASK_FOLDER = "4_answer_MCQA_Qwen2ForCausalLM_answer_pointer"
MODEL_UNIT = "ResidualStream(Layer_0,Token_last_token)"


def _write_identity_checkpoint(path: Path) -> None:
    model_info = {
        "featurizer_class": "IdentityFeaturizerModule",
        "inverse_featurizer_class": "IdentityInverseFeaturizerModule",
        "n_features": None,
        "additional_config": {},
        "featurizer_id": "aana_identity_control",
    }
    torch.save({"model_info": model_info, "state_dict": {}}, path)


def generate_submission(output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    task_dir = output_dir / TASK_FOLDER
    task_dir.mkdir(parents=True)

    base = task_dir / MODEL_UNIT
    _write_identity_checkpoint(Path(f"{base}_featurizer"))
    _write_identity_checkpoint(Path(f"{base}_inverse_featurizer"))
    Path(f"{base}_indices").write_text("null", encoding="utf-8")

    manifest = {
        "method": "AANA identity/control baseline",
        "benchmark": "MIB causal-variable localization track",
        "task_folder": TASK_FOLDER,
        "model_unit": MODEL_UNIT,
        "aana_components": {
            "f_theta": "MIB task model evaluated by the benchmark",
            "E_phi": [
                "submission-shape verifier",
                "identity invertibility verifier",
                "manifest/result-boundary verifier",
            ],
            "R": "MIB public submission contract and generated checkpoint metadata",
            "Pi_psi": "accept verified identity baseline; otherwise regenerate or refuse submission",
            "G": "MIB verify_submission.py",
        },
        "claim_boundary": (
            "Format-valid identity/control baseline only. This is not a "
            "leaderboard score or a learned mechanistic localization method."
        ),
    }
    (output_dir / "aana_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="aana_identity_submission", type=Path)
    args = parser.parse_args()
    generate_submission(args.output_dir)
    print(f"Wrote AANA identity baseline submission to {args.output_dir}")


if __name__ == "__main__":
    main()
