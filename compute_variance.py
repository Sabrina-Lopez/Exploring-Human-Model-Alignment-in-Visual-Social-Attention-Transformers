import wandb
import numpy as np
from collections import defaultdict
from dotenv import load_dotenv
import os

load_dotenv()
wandb.login(key=os.environ.get("WANDB_API_KEY"))

api = wandb.Api()

# REPLACE THESE VARIABLES WITH VALUES THAT MATCH YOUR TRAINING AND TESTING WANDB PROJECT PATHS
train_project = "username/current-wandb-project-title-for-training"
test_project = "username/current-wandb-project-title-for-testing"


def extract_test_base_name(model_name):
    return model_name.rsplit("_", 1)[0]


def compute_variance_summary(groups):
    summary = []
    
    for key, values in groups.items():
        np_vals = np.array(values, dtype=float)
        if len(np_vals) > 1:
            mean = float(np.mean(np_vals))
            var  = float(np.var(np_vals, ddof=1))   # Sample variance
            std  = float(np.std(np_vals, ddof=1))   # Sample standard deviation
            summary.append((key, mean, std, var))

    # Sort by mean, descending 
    return sorted(summary, key=lambda x: x[1], reverse=True)


def print_grouped_variance(title, summary):
    print(f"=== {title.upper()} ===")
    if not summary:
        print("(no groups with > 1 runs)")
    for key, mean, std, var in summary:
        # Now also prints sample standard deviation
        print(f"{key}: Mean = {mean:.5f}, Std = {std:.6f}, Variance = {var:.6f}")
    print()


# Training and validation variance
runs = api.runs(train_project)
grouped_train = defaultdict(list)
grouped_val = defaultdict(list)

for run in runs:
    config = run.config

    if all(k in config for k in ["model_name", "batch_size", "attn_dropout", "hidden_dropout", "lr", "w_decay"]):
        model_name = config["model_name"]

        model_type = None
        if "vivit" in model_name.lower():
            model_type = "vivit"
        elif "timesformer" in model_name.lower():
            model_type = "timesformer"
        elif "vjepa2" in model_name.lower():
            model_type = "vjepa2"
        else:
            continue

        group_key = (
            model_type,
            model_name,
            config["batch_size"],
            config["attn_dropout"],
            config["hidden_dropout"],
            config["lr"],
            config["w_decay"]
        )

        # Pull the last recorded value for each metric (end of epoch)
        history = run.history(keys=["train_accuracy", "val_accuracy"])

        if not history.empty:
            if "train_accuracy" in history.columns:
                train_val = history["train_accuracy"].dropna().values
                if len(train_val) > 0:
                    grouped_train[group_key].append(train_val[-1])
            if "val_accuracy" in history.columns:
                val_val = history["val_accuracy"].dropna().values
                if len(val_val) > 0:
                    grouped_val[group_key].append(val_val[-1])

# Sort and print training/validation stats
train_vivit = compute_variance_summary({k: v for k, v in grouped_train.items() if k[0] == "vivit"})
train_times = compute_variance_summary({k: v for k, v in grouped_train.items() if k[0] == "timesformer"})
train_vjepa2 = compute_variance_summary({k: v for k, v in grouped_train.items() if k[0] == "vjepa2"})
val_vivit = compute_variance_summary({k: v for k, v in grouped_val.items() if k[0] == "vivit"})
val_times = compute_variance_summary({k: v for k, v in grouped_val.items() if k[0] == "timesformer"})
val_vjepa2 = compute_variance_summary({k: v for k, v in grouped_val.items() if k[0] == "vjepa2"})

print_grouped_variance("ViViT Train Accuracy", train_vivit)
print_grouped_variance("ViViT Validation Accuracy", val_vivit)
print_grouped_variance("TimeSformer Train Accuracy", train_times)
print_grouped_variance("TimeSformer Validation Accuracy", val_times)
print_grouped_variance("VJEPA2 Train Accuracy", train_vjepa2)
print_grouped_variance("VJEPA2 Validation Accuracy", val_vjepa2)


# Testing variance (accuracy + per-class recall variance)

test_runs = api.runs(test_project)

# Accuracy grouping (existing)
grouped_test_acc = defaultdict(list)

# Per-class recall groupings
grouped_test_recall = {
    "help": defaultdict(list),
    "hinder": defaultdict(list),
    "physical": defaultdict(list),
}

RECALL_KEYS = ["recall_help", "recall_hinder", "recall_physical"]

for run in test_runs:
    config = run.config
    if "model_name" not in config:
        continue

    model_name = config["model_name"]
    model_type = None
    name_lower = model_name.lower()

    if "vivit" in name_lower:
        model_type = "vivit"
    elif "timesformer" in name_lower:
        model_type = "timesformer"
    elif "vjepa2" in name_lower:
        model_type = "vjepa2"
    else:
        continue

    base_name = extract_test_base_name(model_name)

    # Pull accuracy and recalls; use last available value
    history = run.history(keys=["accuracy"] + RECALL_KEYS)

    # Accuracy
    if not history.empty and "accuracy" in history.columns:
        acc_vals = history["accuracy"].dropna().values
        if len(acc_vals) > 0:
            grouped_test_acc[(model_type, base_name)].append(float(acc_vals[-1]))
    else:
        # Fallback to summary if needed
        acc_sum = run.summary.get("accuracy")
        if isinstance(acc_sum, (int, float)) and not np.isnan(acc_sum):
            grouped_test_acc[(model_type, base_name)].append(float(acc_sum))

    # Per-class recall
    for cls, key in zip(["help", "hinder", "physical"], RECALL_KEYS):
        vals = []
        if not history.empty and key in history.columns:
            vals = history[key].dropna().values

        if len(vals) > 0:
            grouped_test_recall[cls][(model_type, base_name)].append(float(vals[-1]))
        else:
            # Fallback to summary
            s_val = run.summary.get(key)
            if isinstance(s_val, (int, float)) and not np.isnan(s_val):
                grouped_test_recall[cls][(model_type, base_name)].append(float(s_val))

# Accuracy stats
test_vivit_acc = compute_variance_summary({k: v for k, v in grouped_test_acc.items() if k[0] == "vivit"})
test_times_acc = compute_variance_summary({k: v for k, v in grouped_test_acc.items() if k[0] == "timesformer"})
test_vjepa2_acc = compute_variance_summary({k: v for k, v in grouped_test_acc.items() if k[0] == "vjepa2"})

print_grouped_variance("ViViT Test Accuracy", test_vivit_acc)
print_grouped_variance("TimeSformer Test Accuracy", test_times_acc)
print_grouped_variance("VJEPA2 Test Accuracy", test_vjepa2_acc)

# Per-class recall stats and std
for cls in ["help", "hinder", "physical"]:
    vivit_summary = compute_variance_summary({k: v for k, v in grouped_test_recall[cls].items() if k[0] == "vivit"})
    times_summary = compute_variance_summary({k: v for k, v in grouped_test_recall[cls].items() if k[0] == "timesformer"})
    vjepa2_summary = compute_variance_summary({k: v for k, v in grouped_test_recall[cls].items() if k[0] == "vjepa2"})

    print_grouped_variance(f"ViViT Test Recall ({cls})", vivit_summary)
    print_grouped_variance(f"TimeSformer Test Recall ({cls})", times_summary)
    print_grouped_variance(f"VJEPA2 Test Recall ({cls})", vjepa2_summary)