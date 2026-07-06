from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, confusion_matrix, classification_report
)
from src.utils import get_logger, get_device, set_seed

class SentimentClassifier:
    """DistilBERT + LoRA for 3-class sentiment classification."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self.logger = get_logger(__name__, config)
        self.model_cfg = config["models"]["sentiment"]
        self.device = get_device()
        self.logger.info("Device: %s", self.device)
        self.model = None
        self.tokenizer = None

    def build(self) -> None:
        """Load base model and wrap with LoRA adapters."""
        base = self.model_cfg["base_model"]
        num_labels = self.config["sentiment"]["num_classes"]

        self.logger.info("Loading base model: %s", base)
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(base)
        base_model = DistilBertForSequenceClassification.from_pretrained(
            base, num_labels=num_labels
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
        self.logger.info("Model built and moved to %s", self.device)

    def save(self, path: str | Path | None = None) -> Path:
        """Save model weights and tokenizer."""
        save_dir = Path(path or self.model_cfg["save_dir"])
        save_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        self.logger.info("Model saved to %s", save_dir)
        return save_dir

    def load(self, path: str | Path | None = None) -> None:
        """Load saved model and tokenizer from disk."""
        from peft import PeftModel
        load_dir = Path(path or self.model_cfg["save_dir"])
        base = self.model_cfg["base_model"]
        num_labels = self.config["sentiment"]["num_classes"]
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(load_dir)
        base_model = DistilBertForSequenceClassification.from_pretrained(
            base, num_labels=num_labels
        )
        self.model = PeftModel.from_pretrained(base_model, load_dir)
        self.model.to(self.device)
        self.logger.info("Model loaded from %s", load_dir)

    def predict(self, texts: list[str]) -> list[int]:
        """Run inference on a list of texts. Returns label indices."""
        self.model.eval()
        max_len = self.config["data"]["max_text_length"]
        encodings = self.tokenizer(
            texts, max_length=max_len, padding=True,
            truncation=True, return_tensors="pt"
        )
        encodings = {k: v.to(self.device) for k, v in encodings.items()}
        with torch.no_grad():
            outputs = self.model(**encodings)
        return outputs.logits.argmax(dim=-1).cpu().tolist()


class SentimentTrainer:
    """Training loop with early stopping, LR scheduling, and checkpointing."""

    def __init__(self, classifier: SentimentClassifier) -> None:
        self.clf = classifier
        self.config = classifier.config
        self.logger = get_logger(__name__, self.config)
        self.model_cfg = self.config["models"]["sentiment"]
        self.device = classifier.device
        set_seed(self.config["project"]["random_seed"])

    def train(self, train_loader, val_loader) -> dict:
        """Full training loop.

        Returns dict with training history (loss, f1 per epoch).
        """
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
        scheduler = OneCycleLR(
            optimizer, max_lr=lr, total_steps=total_steps, pct_start=0.1
        )

        history = {"train_loss": [], "val_loss": [], "val_f1": []}
        best_f1 = 0.0
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_loss = self._train_epoch(model, train_loader, optimizer, scheduler, max_grad_norm)
            val_loss, val_f1 = self._eval_epoch(model, val_loader)
            elapsed = time.time() - t0

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_f1"].append(val_f1)

            self.logger.info(
                "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | val_f1=%.4f | %.1fs",
                epoch, epochs, train_loss, val_loss, val_f1, elapsed
            )

            # Checkpoint if best
            if val_f1 > best_f1:
                best_f1 = val_f1
                patience_counter = 0
                self.clf.save(save_dir / "best")
                self.logger.info("  New best model saved (f1=%.4f)", best_f1)
            else:
                patience_counter += 1
                self.logger.info(
                    "  No improvement. Patience %d/%d", patience_counter, patience
                )
                if patience_counter >= patience:
                    self.logger.info("Early stopping triggered.")
                    break

            # Save checkpoint every epoch
            ckpt_path = save_dir / f"checkpoint_epoch{epoch}"
            self.clf.save(ckpt_path)

        # Save training history
        with open(save_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)
        self.logger.info("Training complete. Best val F1: %.4f", best_f1)
        return history

    def _train_epoch(self, model, loader, optimizer, scheduler, max_grad_norm) -> float:
        model.train()
        total_loss = 0.0
        for batch in loader:
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
        return total_loss / len(loader)

    def _eval_epoch(self, model, loader) -> tuple[float, float]:
        model.eval()
        total_loss = 0.0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                total_loss += outputs.loss.item()
                preds = outputs.logits.argmax(dim=-1).cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().tolist())
        avg_loss = total_loss / len(loader)
        f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
        return avg_loss, f1


def evaluate_sentiment(classifier: SentimentClassifier, test_loader) -> dict:
    """Run full evaluation on test set. Returns metrics dict."""
    logger = get_logger(__name__, classifier.config)
    label_map = classifier.config["sentiment"]["label_map"]
    id2label = {v: k for k, v in label_map.items()}

    model = classifier.model
    device = classifier.device
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = outputs.logits.argmax(dim=-1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().tolist())

    label_names = [id2label.get(i, str(i)) for i in sorted(set(all_labels + all_preds))]
    metrics = {
        "accuracy": round(accuracy_score(all_labels, all_preds), 4),
        "f1_weighted": round(f1_score(all_labels, all_preds, average="weighted", zero_division=0), 4),
        "f1_macro": round(f1_score(all_labels, all_preds, average="macro", zero_division=0), 4),
        "precision_weighted": round(precision_score(all_labels, all_preds, average="weighted", zero_division=0), 4),
        "recall_weighted": round(recall_score(all_labels, all_preds, average="weighted", zero_division=0), 4),
        "confusion_matrix": confusion_matrix(all_labels, all_preds).tolist(),
        "classification_report": classification_report(
            all_labels, all_preds, target_names=label_names, zero_division=0
        ),
    }
    logger.info("Test accuracy: %.4f | F1 weighted: %.4f", metrics["accuracy"], metrics["f1_weighted"])
    logger.info("\n%s", metrics["classification_report"])
    return metrics
