"""
download_models.py

Downloads pre-trained weights and configurations for all 8 evaluation models
spanning GBDTs, Tabular Deep Learning, and classic baselines to benchmark
against Pandora's Regret.

Usage:
    pip install huggingface_hub
    python download_models.py --output_dir ./models
"""

import argparse
from pathlib import Path
from huggingface_hub import hf_hub_download

# Define all 8 models grouped by their architecture class
MODELS = {
    # ── 1. Gradient Boosted Decision Trees (GBDTs) ───────────────────────────
    "xgb_baseline": {
        "repo_id": "dreamer-labs/ieee-cis-xgboost-baseline",
        "filename": "xgb_model.json",
        "category": "gbdt",
        "desc": "XGBoost binary/multiclass tree checkpoint"
    },
    "lgbm_baseline": {
        "repo_id": "dreamer-labs/ieee-cis-lgbm-baseline",
        "filename": "model.txt",
        "category": "gbdt",
        "desc": "LightGBM booster tree file"
    },
    "catboost_baseline": {
        "repo_id": "dreamer-labs/ieee-cis-catboost-baseline",
        "filename": "catboost_model.bin",
        "category": "gbdt",
        "desc": "CatBoost serialized model"
    },
    # ── 2. Tabular Deep Learning Architectures ────────────────────────────────
    "ft_transformer": {
        "repo_id": "dreamer-labs/ieee-cis-ft-transformer",
        "filename": "pytorch_model.bin",
        "category": "tabular_dl",
        "desc": "Feature Tokenizer Transformer weights"
    },
    "tabnet": {
        "repo_id": "dreamer-labs/ieee-cis-tabnet",
        "filename": "tabnet_network.zip",
        "category": "tabular_dl",
        "desc": "TabNet model zip archive (weights + hyperparameters)"
    },
    "saint_model": {
        "repo_id": "dreamer-labs/ieee-cis-saint",
        "filename": "saint_best_weights.pt",
        "category": "tabular_dl",
        "desc": "SAINT (Self-Attention & Invariant Tabular) checkpoint"
    },
    "node_model": {
        "repo_id": "dreamer-labs/ieee-cis-node",
        "filename": "node_network.pt",
        "category": "tabular_dl",
        "desc": "Neural Oblivious Decision Ensembles weights"
    },
    # ── 3. Classic / Statistical Baselines ───────────────────────────────────
    "logistic_regression": {
        "repo_id": "dreamer-labs/ieee-cis-logistic-regression",
        "filename": "lr_platt_scaled.joblib",
        "category": "classic",
        "desc": "Platt-Scaled Logistic Regression estimator"
    }
}

def download_all_baselines(output_dir: Path):
    """Downloads all 8 tabular architectures from HF Hub."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print(f"🚀 Downloading 8 Baseline Architectures to: {output_dir}")
    print("=" * 70)

    success_count = 0
    for name, info in MODELS.items():
        model_subfolder = output_dir / info["category"] / name
        model_subfolder.mkdir(parents=True, exist_ok=True)
        
        print(f"\n📦 [{info['category'].upper()}] Fetching {name}...")
        print(f"   Description: {info['desc']}")
        
        try:
            # Download weights file from HF Hub
            downloaded_file = hf_hub_download(
                repo_id=info["repo_id"],
                filename=info["filename"],
                local_dir=model_subfolder,
                local_dir_use_symlinks=False
            )
            print(f"   ✅ Successfully downloaded → {Path(downloaded_file).name}")
            success_count += 1
            
        except Exception as e:
            print(f"   ❌ Failed to download {name}.")
            print(f"      Reason: {e}")

    print("\n" + "=" * 70)
    print(f"🎯 Download Process Finished: {success_count}/8 Models Stored.")
    print("=" * 70)
    print(f"File layout on disk:")
    for cat in ["gbdt", "tabular_dl", "classic"]:
        cat_path = output_dir / cat
        if cat_path.exists():
            print(f"  📂 {cat}/")
            for subdir in cat_path.iterdir():
                print(f"    └── 📄 {subdir.name}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download pre-trained weights for all 8 baseline models.")
    parser.add_argument(
        "--output_dir", 
        type=Path, 
        default=Path("./models"),
        help="Target folder for local model weight storage."
    )
    args = parser.parse_args()
    
    download_all_baselines(args.output_dir)
