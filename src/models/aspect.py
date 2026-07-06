from __future__ import annotations
import json
import time
from pathlib import Path
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from transformers import RobertaForSequenceClassification, RobertaTokenizerFast
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import f1_score, hamming_loss, classification_report
from tqdm import tqdm
from src.utils import get_logger, get_device, set_seed


# ── Pseudo-labeling keyword rules ────────────────────────────────────────────
ASPECT_KEYWORDS = {
    "price": [
        "price", "cost", "expensive", "cheap", "affordable", "value",
        "worth", "overpriced", "money", "dollar", "fee", "budget",
    ],
    "quality": [
        "quality", "durable", "sturdy", "flimsy", "broke", "material",
        "build", "construction", "solid", "poor quality", "well made",
        "cheaply made", "fell apart", "lasted",
    ],
    "delivery": [
        "shipping", "delivery", "arrived", "package", "transit", "days",
        "late", "early", "fast shipping", "slow delivery", "damaged in transit",
        "box", "arrived broken",
    ],
    "customer_service": [
        "customer service", "support", "return", "refund", "exchange",
        "representative", "contact", "responded", "helpful staff",
        "rude", "service team",
    ],
    "packaging": [
        "packaging", "packed", "wrapped", "box", "bubble wrap",
        "container", "opened", "sealed", "damaged packaging",
    ],
    "usability": [
        "easy to use", "instructions", "setup", "install", "difficult",
        "intuitive", "confusing", "user friendly", "works great",
        "does what", "as described", "simple", "complicated",
    ],
}


def generate_aspect_labels(df, aspects):
    """Generate binary aspect labels from review text using keyword matching.

    This is weak/distant supervision — a legitimate technique when manual
    labels are unavailable. Documents well in a portfolio.

    Args:
        df: DataFrame with 'text' column.
        aspects: List of aspect names matching ASPECT_KEYWORDS keys.

    Returns:
        DataFrame with new binary columns for each aspect.
    """
    import pandas as pd
    df = df.copy()
    for aspect in aspects:
        keywords = ASPECT_KEYWORDS.get(aspect, [])
        pattern = "|".join(keywords)
        df[aspect] = df["text"].str.lower().str.contains(pattern, regex=True).astype(int)
    return df


class AspectClassifier:
    """RoBERTa + LoRA for multi-label aspect classification."""

    def __init__(self, config):
        self.config = config
        self.logger = get_logger(__name__, config)
        self.model_cfg = config["models"]["aspect"]
        self.aspects = self.model_cfg["aspects"]
        self.num_aspects = len(self.aspects)
        self.device = get_device()
        self.logger.info("Device: %s", self.device)
        self.model = None
        self.tokenizer = None

    def build(self):
        base = self.model_cfg["base_model"]
        self.logger.info("Loading base model: %s", base)
        self.tokenizer = RobertaTokenizerFast.from_pretrained(base)
        base_model = RobertaForSequenceClassification.from_pretrained(
            base,
            num_labels=self.num_aspects,
            problem_type="multi_label_classification",
        )
        lora_cfg = self.model_cfg["lora"]
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=lora_cfg["r"],
            lora_alpha=lora_cfg["lora_alpha"],
            lora_dropout=lora_cfg["lora_dropout"],
            target_modules=lora_cfg["target_modules"],
            bias="none",
        )
        self.model = get_peft_model(base_model, peft_config)
        self.model.print_trainable_parameters()
        self.model.to(self.device)
        self.logger.info("Model built on %s", self.device)

    def save(self, path=None):
        save_dir = Path(path or self.model_cfg["save_dir"])
        save_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        self.logger.info("Saved -> %s", save_dir)
        return save_dir

    def load(self, path=None):
        from peft import PeftModel
        load_dir = Path(path or self.model_cfg["save_dir"])
        base = self.model_cfg["base_model"]
        self.tokenizer = RobertaTokenizerFast.from_pretrained(load_dir)
        base_model = RobertaForSequenceClassification.from_pretrained(
            base,
            num_labels=self.num_aspects,
            problem_type="multi_label_classification",
        )
        self.model = PeftModel.from_pretrained(base_model, load_dir)
        self.model.to(self.device)
        self.logger.info("Loaded from %s", load_dir)

    def predict(self, texts, threshold=0.5):
        self.model.eval()
        max_len = self.config["data"]["max_text_length"]
        encodings = self.tokenizer(
            texts, max_length=max_len, padding=True,
            truncation=True, return_tensors="pt"
        )
        encodings = {k: v.to(self.device) for k, v in encodings.items()}
        with torch.no_grad():
            outputs = self.model(**encodings)
        probs = torch.sigmoid(outputs.logits).cpu().numpy()
        return (probs >= threshold).astype(int).tolist()


class AspectTrainer:
    """Training loop for multi-label aspect classification."""

    def __init__(self, classifier):
        self.clf = classifier
        self.config = classifier.config
        self.logger = get_logger(__name__, self.config)
        self.model_cfg = self.config["models"]["aspect"]
        self.device = classifier.device
        set_seed(self.config["project"]["random_seed"])

    def train(self, train_loader, val_loader):
        model = self.clf.model
        epochs = self.model_cfg["epochs"]
        lr = self.model_cfg["learning_rate"]
        weight_decay = self.model_cfg["weight_decay"]
        max_grad_norm = self.model_cfg["max_grad_norm"]
        patience = self.model_cfg["early_stopping_patience"]
        save_dir = Path(self.model_cfg["save_dir"])
        save_dir.mkdir(parents=True, exist_ok=True)

        optimizer = AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr, weight_decay=weight_decay
        )
        total_steps = len(train_loader) * epochs
        scheduler = OneCycleLR(optimizer, max_lr=lr, total_steps=total_steps, pct_start=0.1)

        history = {"train_loss": [], "val_loss": [], "val_f1_micro": []}
        best_f1 = 0.0
        patience_counter = 0

        self.logger.info("Training config: epochs=%d lr=%s batch=%d steps_per_epoch=%d",
            epochs, lr, self.model_cfg["batch_size"], len(train_loader))

        for epoch in range(1, epochs + 1):
            self.logger.info("")
            self.logger.info("=" * 60)
            self.logger.info("EPOCH %d / %d  (patience=%d/%d  best_f1=%.4f)",
                epoch, epochs, patience_counter, patience, best_f1)
            self.logger.info("=" * 60)
            t0 = time.time()

            train_loss = self._train_epoch(model, train_loader, optimizer, scheduler, max_grad_norm, epoch, epochs)
            self.logger.info("Validation pass starting ...")
            val_loss, val_f1_micro, val_f1_macro = self._eval_epoch(model, val_loader)
            elapsed = time.time() - t0

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_f1_micro"].append(val_f1_micro)

            self.logger.info("")
            self.logger.info("Epoch %d/%d Summary:", epoch, epochs)
            self.logger.info("  train_loss  : %.4f", train_loss)
            self.logger.info("  val_loss    : %.4f", val_loss)
            self.logger.info("  val_f1_micro: %.4f  (best=%.4f)", val_f1_micro, best_f1)
            self.logger.info("  val_f1_macro: %.4f", val_f1_macro)
            self.logger.info("  time        : %.1fs  (%.1f min)", elapsed, elapsed / 60)

            if val_f1_micro > best_f1:
                best_f1 = val_f1_micro
                patience_counter = 0
                self.clf.save(save_dir / "best")
                self.logger.info("  *** New best model saved (f1_micro=%.4f) ***", best_f1)
            else:
                patience_counter += 1
                self.logger.info("  No improvement. Patience %d/%d", patience_counter, patience)
                if patience_counter >= patience:
                    self.logger.info("Early stopping at epoch %d.", epoch)
                    break

            self.clf.save(save_dir / f"checkpoint_epoch{epoch}")
            self.logger.info("  Checkpoint saved.")

        with open(save_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)
        self.logger.info("Training complete. Best val F1 micro: %.4f", best_f1)
        return history

    def _train_epoch(self, model, loader, optimizer, scheduler, max_grad_norm, epoch, total_epochs):
        model.train()
        total_loss = 0.0
        n_batches = len(loader)
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [train]", unit="batch", ncols=110, leave=True)

        for step, batch in enumerate(pbar, 1):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            avg_loss = total_loss / step
            current_lr = scheduler.get_last_lr()[0]
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "avg": f"{avg_loss:.4f}", "lr": f"{current_lr:.2e}"})

            if step % 50 == 0 or step == n_batches:
                self.logger.info("  [Epoch %d | Step %d/%d] loss=%.4f avg=%.4f lr=%.2e",
                    epoch, step, n_batches, loss.item(), avg_loss, current_lr)

        return total_loss / n_batches

    def _eval_epoch(self, model, loader):
        model.eval()
        total_loss = 0.0
        all_preds, all_labels = [], []
        pbar = tqdm(loader, desc="Validation", unit="batch", ncols=110, leave=False)

        with torch.no_grad():
            for batch in pbar:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                total_loss += outputs.loss.item()
                probs = torch.sigmoid(outputs.logits).cpu().numpy()
                preds = (probs >= 0.5).astype(int).tolist()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy().tolist())

        avg_loss = total_loss / len(loader)
        f1_micro = f1_score(all_labels, all_preds, average="micro", zero_division=0)
        f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        return avg_loss, f1_micro, f1_macro


def evaluate_aspect(classifier, test_loader):
    logger = get_logger(__name__, classifier.config)
    aspects = classifier.aspects
    model = classifier.model
    device = classifier.device
    model.eval()

    all_preds, all_labels = [], []
    pbar = tqdm(test_loader, desc="Testing aspects", unit="batch", ncols=110)

    with torch.no_grad():
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.sigmoid(outputs.logits).cpu().numpy()
            preds = (probs >= 0.5).astype(int).tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy().tolist())

    metrics = {
        "f1_micro": round(f1_score(all_labels, all_preds, average="micro", zero_division=0), 4),
        "f1_macro": round(f1_score(all_labels, all_preds, average="macro", zero_division=0), 4),
        "hamming_loss": round(hamming_loss(all_labels, all_preds), 4),
        "per_aspect_f1": {},
    }
    per_aspect = f1_score(all_labels, all_preds, average=None, zero_division=0)
    for i, aspect in enumerate(aspects):
        metrics["per_aspect_f1"][aspect] = round(float(per_aspect[i]), 4)
        logger.info("  %-20s F1: %.4f", aspect, per_aspect[i])

    logger.info("Aspect F1 micro=%.4f  macro=%.4f  hamming=%.4f",
        metrics["f1_micro"], metrics["f1_macro"], metrics["hamming_loss"])
    return metrics
