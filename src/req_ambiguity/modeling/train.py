"""Train DeBERTa multi-label ambiguity classifier from train.yaml."""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import DebertaV2Tokenizer, get_linear_schedule_with_warmup

from req_ambiguity.evaluation.metrics import multilabel_metrics, find_optimal_threshold
from req_ambiguity.modeling.classifier import DeBERTaAmbiguityClassifier
from req_ambiguity.preprocessing.tokenize import UserStoryDataset
from req_ambiguity.utils.config import resolve_path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
        "text": [b["text"] for b in batch],
    }


def _resolve_train_csv(cfg: Mapping[str, Any], *, project_root: Path) -> Path:
    paths = cfg["paths"]
    use_aug = cfg.get("data", {}).get("use_augmented_data", False)
    if use_aug:
        p = resolve_path(paths["augmented_dir"], root=project_root) / "train_augmented.csv"
        hint = "Run scripts/augment.py (or --force) to create data/augmented/train_augmented.csv."
    else:
        p = resolve_path(paths["processed_dir"], root=project_root) / "train.csv"
        hint = "Run scripts/preprocess.py first."
    if not p.is_file():
        raise FileNotFoundError(f"Missing training CSV: {p}. {hint}")
    return p


def _load_split_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path, encoding="utf-8")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    loss_fn: nn.Module,
    device: torch.device,
    max_grad_norm: float = 1.0,
    *,
    epoch: int,
    total_epochs: int,
    show_progress: bool = True,
    use_amp: bool = False,
    scaler: Any = None,
    accumulation_steps: int = 1,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    pbar = tqdm(
        loader,
        desc=f"Epoch {epoch}/{total_epochs} [train]",
        leave=True,
        disable=not show_progress,
    )
    optimizer.zero_grad(set_to_none=True)
    for i, batch in enumerate(pbar):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits, labels)
            loss = loss / accumulation_steps
            
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
            
        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(loader):
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            
        loss_val = float(loss.detach().cpu()) * accumulation_steps
        total_loss += loss_val
        n_batches += 1
        postfix: dict[str, Any] = {"loss": f"{loss_val:.4f}"}
        if scheduler is not None:
            postfix["lr"] = f"{scheduler.get_last_lr()[0]:.2e}"
        pbar.set_postfix(postfix)
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    *,
    desc: str = "Eval",
    show_progress: bool = True,
    use_amp: bool = False,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    logits_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []
    for batch in tqdm(loader, desc=desc, leave=False, disable=not show_progress):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits, labels)
        total_loss += float(loss.detach().cpu())
        n_batches += 1
        logits_list.append(logits.detach().cpu().numpy())
        labels_list.append(labels.detach().cpu().numpy())
    y_logits = np.concatenate(logits_list, axis=0) if logits_list else np.zeros((0, 0))
    y_true = np.concatenate(labels_list, axis=0) if labels_list else np.zeros((0, 0))
    return total_loss / max(n_batches, 1), y_true, y_logits


def _metric_value(metrics: Mapping[str, Any], name: str) -> float:
    if name == "macro_f1":
        return float(metrics["macro_f1"])
    if name == "micro_f1":
        return float(metrics["micro_f1"])
    raise ValueError(f"Unknown best_model_metric: {name!r} (use macro_f1 or micro_f1)")


def _plot_history(history: list[dict[str, Any]], figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    val_macro = [h["val_macro_f1"] for h in history]
    val_micro = [h["val_micro_f1"] for h in history]

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_loss, label="train_loss")
    plt.plot(epochs, val_loss, label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "loss_curve.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, val_macro, label="val_macro_f1")
    plt.plot(epochs, val_micro, label="val_micro_f1")
    plt.xlabel("epoch")
    plt.ylabel("F1")
    plt.ylim(0.0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "f1_curve.png", dpi=150)
    plt.close()


def _plot_roc_curves(y_true: np.ndarray, probs: np.ndarray, label_names: list[str], figures_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, auc
    except ImportError:
        return
    plt.figure(figsize=(10, 8))
    for i, name in enumerate(label_names):
        yt = y_true[:, i]
        pr = probs[:, i]
        if np.unique(yt).size >= 2:
            fpr, tpr, _ = roc_curve(yt, pr)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves per Ambiguity Type")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(figures_dir / "roc_curves.png", dpi=150)
    plt.close()


def _plot_pr_curves(y_true: np.ndarray, probs: np.ndarray, label_names: list[str], figures_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import precision_recall_curve, average_precision_score
    except ImportError:
        return
    plt.figure(figsize=(10, 8))
    for i, name in enumerate(label_names):
        yt = y_true[:, i]
        pr = probs[:, i]
        if np.unique(yt).size >= 2:
            precision, recall, _ = precision_recall_curve(yt, pr)
            ap = average_precision_score(yt, pr)
            plt.plot(recall, precision, lw=2, label=f"{name} (AP = {ap:.2f})")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves per Ambiguity Type")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(figures_dir / "pr_curves.png", dpi=150)
    plt.close()


def _plot_confusion_matrices(y_true: np.ndarray, preds: np.ndarray, label_names: list[str], figures_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    except ImportError:
        return
    n_labels = len(label_names)
    cols = 3
    rows = (n_labels + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = axes.flatten()
    for i, name in enumerate(label_names):
        yt = y_true[:, i]
        yp = preds[:, i]
        cm = confusion_matrix(yt, yp)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No", "Yes"])
        disp.plot(ax=axes[i], cmap='Blues', values_format='d')
        axes[i].set_title(name)
    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.savefig(figures_dir / "confusion_matrices.png", dpi=150)
    plt.close()


def _log(msg: str) -> None:
    print(msg, flush=True)


def train_from_config(
    cfg: Mapping[str, Any], *, project_root: Path, show_progress: bool = True, save_artifacts: bool = True
) -> dict[str, Any]:
    paths = cfg["paths"]
    # Writable HF cache (CI/sandbox may block ~/.cache; force project-local cache)
    hf_home = (project_root / ".hf_cache").resolve()
    hf_home.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_HUB_CACHE"] = str(hf_home / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(hf_home / "transformers")
    mpl_dir = (project_root / ".mplconfig").resolve()
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_dir)
    
    t_start = time.perf_counter()

    label_cols: list[str] = list(cfg["label_cols"])
    text_column = str(paths["text_column"])
    model_name = str(cfg["model_name"])
    max_length = int(cfg["max_length"])
    batch_size = int(cfg.get("batch_size", 16))
    accumulation_steps = int(cfg.get("gradient_accumulation_steps", 1))
    use_amp = bool(cfg.get("use_mixed_precision", True))
    num_workers = int(cfg.get("dataloader_num_workers", 2))
    lr = float(cfg["learning_rate"])
    weight_decay = float(cfg["weight_decay"])
    epochs = int(cfg["epochs"])
    warmup_ratio = float(cfg["warmup_ratio"])
    dropout = float(cfg["dropout_rate"])
    seed = int(cfg["random_seed"])
    best_metric_name = str(cfg.get("best_model_metric", "macro_f1"))

    processed_dir = resolve_path(paths["processed_dir"], root=project_root)
    checkpoints_dir = resolve_path(paths["checkpoints_dir"], root=project_root)
    best_ckpt_path = resolve_path(paths["best_checkpoint"], root=project_root)
    logs_dir = resolve_path(paths["training_logs_dir"], root=project_root)
    figures_dir = resolve_path(paths["figures_dir"], root=project_root) / "training"

    set_seed(seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    scaler = torch.amp.GradScaler('cuda') if (use_amp and device.type == "cuda") else None

    train_csv = _resolve_train_csv(cfg, project_root=project_root)
    val_csv = processed_dir / "val.csv"
    test_csv = processed_dir / "test.csv"

    df_train = _load_split_csv(train_csv)
    df_val = _load_split_csv(val_csv)
    df_test = _load_split_csv(test_csv)

    tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)
    train_ds = UserStoryDataset.from_dataframe(
        df_train, text_column=text_column, label_cols=label_cols, tokenizer=tokenizer, max_length=max_length
    )
    val_ds = UserStoryDataset.from_dataframe(
        df_val, text_column=text_column, label_cols=label_cols, tokenizer=tokenizer, max_length=max_length
    )
    test_ds = UserStoryDataset.from_dataframe(
        df_test, text_column=text_column, label_cols=label_cols, tokenizer=tokenizer, max_length=max_length
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_batch,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch, 
        num_workers=num_workers, pin_memory=(device.type == "cuda"), persistent_workers=(num_workers > 0)
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch, 
        num_workers=num_workers, pin_memory=(device.type == "cuda"), persistent_workers=(num_workers > 0)
    )

    num_labels = len(label_cols)
    model = DeBERTaAmbiguityClassifier(model_name, num_labels=num_labels, dropout=dropout).to(device)
    model = model.float()  # Force FP32 to prevent Half-precision gradient overflow
    
    # Calculate pos_weight for severe class imbalance
    y_train = df_train[label_cols].values
    pos_counts = y_train.sum(axis=0)
    neg_counts = len(df_train) - pos_counts
    pos_counts = np.maximum(pos_counts, 1)  # avoid division by zero
    
    # Mathematical rationale: BCEWithLogitsLoss applies pos_weight to the positive class.
    # For highly imbalanced classes (e.g., TechnicalAmbiguity at 0.3%), raw pos_weight
    # can exceed 300, leading to gradient explosion and NaN loss. 
    pos_weight_strategy = str(cfg.get("loss", {}).get("pos_weight_strategy", "cap")).lower()
    pos_weight_cap = float(cfg.get("loss", {}).get("pos_weight_cap", 50.0))
    raw_weights = neg_counts / pos_counts
    
    if pos_weight_strategy == "sqrt":
        final_weights = np.sqrt(raw_weights)
    else:
        final_weights = np.clip(raw_weights, 1.0, pos_weight_cap)
        
    pos_weight = torch.tensor(final_weights, dtype=torch.float32).to(device)

    # Log pos_weights for thesis
    results_dir = resolve_path(paths.get("results_dir", "outputs/results"), root=project_root)
    results_dir.mkdir(parents=True, exist_ok=True)
    with (results_dir / "pos_weights.txt").open("w", encoding="utf-8") as f:
        f.write(f"Class pos_weights ({pos_weight_strategy}):\n")
        for name, w, raw in zip(label_cols, final_weights, raw_weights):
            f.write(f"{name}: {w:.2f} (raw: {raw:.2f})\n")

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    opt_name = str(cfg.get("optimizer", "adamw")).lower()
    if opt_name != "adamw":
        raise ValueError(f"Unsupported optimizer: {opt_name!r} (only adamw implemented)")
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    early_stop = bool(cfg.get("early_stopping", True))
    patience = int(cfg.get("early_stopping_patience", 3))
    min_delta = float(cfg.get("early_stopping_min_delta", 0.0))

    total_steps = max(len(train_loader) * epochs, 1)
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    _log("=" * 72)
    _log("Training")
    _log(f"  project_root:      {project_root.resolve()}")
    _log(f"  model:             {model_name}")
    _log(f"  device:            {device}")
    if device.type == "cuda":
        _log(f"  cuda_device:       {torch.cuda.get_device_name(torch.cuda.current_device())}")
    _log(f"  torch:             {torch.__version__}")
    _log(f"  rows:              train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")
    _log(f"  batches/epoch:     {len(train_loader)}  batch_size={batch_size}")
    _log(f"  epochs (max):      {epochs}  lr={lr:g}  warmup_ratio={warmup_ratio}")
    _log(f"  early_stopping:    {early_stop}  patience={patience}  min_delta={min_delta}")
    _log(f"  best_model_metric: {best_metric_name}")
    _log(f"  labels ({num_labels}): {', '.join(label_cols)}")
    _log(f"  checkpoint:        {best_ckpt_path}")
    _log(f"  logs:              {logs_dir}")
    _log(f"  figures:           {figures_dir}")
    _log("=" * 72)

    history: list[dict[str, Any]] = []
    best_score = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    stall = 0

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            loss_fn,
            device,
            epoch=epoch,
            total_epochs=epochs,
            show_progress=show_progress,
            use_amp=use_amp,
            scaler=scaler,
            accumulation_steps=accumulation_steps,
        )
        train_s = time.perf_counter() - t0
        t1 = time.perf_counter()
        val_loss, y_true_v, logits_v = evaluate_epoch(
            model,
            val_loader,
            loss_fn,
            device,
            desc=f"Epoch {epoch}/{epochs} [val]",
            show_progress=show_progress,
            use_amp=use_amp,
        )
        val_s = time.perf_counter() - t1
        val_metrics = multilabel_metrics(y_true_v, logits_v, label_names=label_cols, threshold=0.5)
        score = _metric_value(val_metrics, best_metric_name)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_macro_f1": val_metrics["macro_f1"],
            "val_micro_f1": val_metrics["micro_f1"],
            "val_macro_precision": val_metrics["macro_precision"],
            "val_macro_recall": val_metrics["macro_recall"],
        }
        for lbl in label_cols:
            row[f"val_f1_{lbl}"] = val_metrics["per_label"][lbl]["f1"]
            row[f"val_prec_{lbl}"] = val_metrics["per_label"][lbl]["precision"]
            row[f"val_rec_{lbl}"] = val_metrics["per_label"][lbl]["recall"]
        history.append(row)

        improved = score > best_score + min_delta
        if improved:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            stall = 0
        else:
            stall += 1

        mark = "  *** new best ***" if improved else ""
        _log(
            f"Epoch {epoch}/{epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}  val_micro_f1={val_metrics['micro_f1']:.4f}  "
            f"val_P={val_metrics['macro_precision']:.4f}  val_R={val_metrics['macro_recall']:.4f}  "
            f"{best_metric_name}={score:.4f}  best={best_score:.4f}@ep{best_epoch}  "
            f"stall={stall}/{patience}  ({train_s:.0f}s train, {val_s:.0f}s val){mark}"
        )
        
        per_label_strs = [f"{lbl[:4]}={val_metrics['per_label'][lbl]['f1']:.3f}" for lbl in label_cols]
        _log(f"    Per-label F1: {', '.join(per_label_strs)}")

        if early_stop and stall >= patience:
            _log(f"Early stopping: no {best_metric_name} improvement for {patience} epoch(s).")
            with (results_dir / "early_stopping_epoch.txt").open("w", encoding="utf-8") as f:
                f.write(f"Early stopping triggered at epoch: {epoch}\n")
                f.write(f"Best epoch was: {best_epoch}\n")
            break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        best_epoch = epochs

    _log("-" * 72)
    _log(f"Restoring best weights: epoch {best_epoch}  {best_metric_name}={best_score:.4f}")
    _log("-" * 72)

    model.load_state_dict(best_state)
    
    if save_artifacts:
        metadata = {
            "model_name": model_name,
            "label_cols": label_cols,
            "max_length": max_length,
            "num_labels": num_labels,
            "best_epoch": best_epoch,
            "best_val_score": float(best_score),
            "best_model_metric": best_metric_name,
            "train_csv": str(train_csv),
            "random_seed": seed,
            "total_training_time_seconds": float(time.perf_counter() - t_start),
        }
        
        best_ckpt_path = save_best_checkpoint(best_state, checkpoints_dir, metadata)

        tok_dir = checkpoints_dir / "tokenizer"
        tokenizer.save_pretrained(tok_dir)

        logs_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(history).to_csv(logs_dir / "history.csv", index=False)
        with (logs_dir / "history.json").open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        _plot_history(history, figures_dir)

    _log("Sweeping thresholds on validation set...")
    val_loss_final, y_true_v_final, logits_v_final = evaluate_epoch(
        model, val_loader, loss_fn, device, desc="Val Threshold", show_progress=False, use_amp=use_amp
    )
    probs_v_final = 1.0 / (1.0 + np.exp(-logits_v_final))
    
    # Global threshold sweep for plotting and global baseline
    thresholds = np.arange(0.10, 0.92, 0.02)
    global_f1s = []
    from sklearn.metrics import f1_score
    for t in thresholds:
        preds = (probs_v_final >= t).astype(np.int64)
        global_f1s.append(f1_score(y_true_v_final, preds, average="macro", zero_division=0))
    
    best_global_idx = np.argmax(global_f1s)
    optimal_global_threshold = float(thresholds[best_global_idx])
    
    if save_artifacts:
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(8, 5))
            plt.plot(thresholds, global_f1s, marker='o')
            plt.axvline(optimal_global_threshold, color='red', linestyle='--', label=f'Optimal: {optimal_global_threshold:.2f}')
            plt.title("Macro F1 vs. Decision Threshold")
            plt.xlabel("Threshold")
            plt.ylabel("Macro F1 (Validation)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(figures_dir / "threshold_curve.png", dpi=150)
            plt.close()
        except ImportError:
            pass

        with (results_dir / "optimal_global_threshold.txt").open("w", encoding="utf-8") as f:
            f.write(str(optimal_global_threshold))

    # Per-label threshold sweep
    per_label_thresholds = {}
    for i, name in enumerate(label_cols):
        best_l_f1 = -1.0
        best_l_t = 0.5
        yt = y_true_v_final[:, i]
        pr = probs_v_final[:, i]
        for t in thresholds:
            yp = (pr >= t).astype(np.int64)
            f1 = f1_score(yt, yp, zero_division=0)
            if f1 > best_l_f1:
                best_l_f1 = f1
                best_l_t = float(t)
        per_label_thresholds[name] = best_l_t
    
    if save_artifacts:
        with (results_dir / "optimal_thresholds.json").open("w", encoding="utf-8") as f:
            json.dump(per_label_thresholds, f, indent=2)

    _log(f"Optimal Global Threshold: {optimal_global_threshold:.2f}")

    _log("Evaluating on test split with global threshold…")
    test_loss, y_true_t, logits_t = evaluate_epoch(
        model, test_loader, loss_fn, device, desc="Test", show_progress=show_progress, use_amp=use_amp
    )
    test_metrics = multilabel_metrics(y_true_t, logits_t, label_names=label_cols, threshold=optimal_global_threshold)
    _log(
        f"Test  loss={test_loss:.4f}  macro_f1={test_metrics['macro_f1']:.4f}  "
        f"micro_f1={test_metrics['micro_f1']:.4f}  "
        f"macro_P={test_metrics['macro_precision']:.4f}  macro_R={test_metrics['macro_recall']:.4f}"
    )
    test_out = {
        "test_loss": test_loss,
        "metrics": test_metrics,
        "optimal_global_threshold": optimal_global_threshold,
        "best_epoch": best_epoch,
        "best_val_score": float(best_score),
    }
    
    if save_artifacts:
        with (logs_dir / "test_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(test_out, f, indent=2, default=str)

        probs = 1.0 / (1.0 + np.exp(-logits_t))
        pred_df = pd.DataFrame(
            {
                text_column: df_test[text_column].astype(str).values[: len(df_test)],
            }
        )
        for i, c in enumerate(label_cols):
            pred_df[f"{c}_true"] = y_true_t[:, i]
            pred_df[f"{c}_prob"] = probs[:, i]
        pred_df.to_csv(logs_dir / "test_probabilities.csv", index=False)

        np.savez_compressed(
            logs_dir / "test_logits_labels.npz",
            logits=logits_t.astype(np.float32),
            labels=y_true_t.astype(np.float32),
        )

    if save_artifacts:
        _log("=" * 72)
        _log("Saved")
        _log(f"  {best_ckpt_path}")
        _log(f"  {tok_dir}")
        _log(f"  {logs_dir / 'history.csv'}")
        _log(f"  {logs_dir / 'test_metrics.json'}")
        _log(f"  {figures_dir}")
        _log("=" * 72)

    return {
        "best_checkpoint": str(best_ckpt_path) if save_artifacts else None,
        "best_epoch": best_epoch,
        "best_val_score": float(best_score),
        "best_model_metric": best_metric_name,
        "test_macro_f1": test_metrics["macro_f1"],
        "test_micro_f1": test_metrics["micro_f1"],
        "training_logs_dir": str(logs_dir),
        "figures_dir": str(figures_dir),
        "history": history,
    }
