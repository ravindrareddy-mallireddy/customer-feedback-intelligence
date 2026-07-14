from __future__ import annotations
import json
import time
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import f1_score, hamming_loss
from tqdm import tqdm
from src.utils import get_logger, get_device, set_seed

ASPECT_KEYWORDS = {
    "price": ["price", "cost", "expensive", "cheap", "affordable", "value",
              "worth", "overpriced", "money", "dollar", "fee", "budget"],
    "quality": ["quality", "durable", "sturdy", "flimsy", "broke", "material",
                "build", "solid", "poor quality", "well made", "cheaply made",
                "fell apart", "lasted"],
    "delivery": ["shipping", "delivery", "arrived", "package", "transit",
                 "late", "early", "fast shipping", "slow delivery", "days to arrive"],
    "customer_service": ["customer service", "support", "return", "refund",
                         "exchange", "representative", "contact", "responded"],
    "packaging": ["packaging", "packed", "wrapped", "box", "bubble wrap",
                  "container", "sealed", "damaged packaging"],
    "usability": ["easy to use", "instructions", "setup", "install", "difficult",
                  "intuitive", "confusing", "user friendly", "works great",
                  "as described", "simple", "complicated"],
}

def generate_aspect_labels(df, aspects):
    df = df.copy()
    for aspect in aspects:
        keywords = ASPECT_KEYWORDS.get(aspect, [])
        pattern = "|".join(keywords)
        df[aspect] = df["text"].str.lower().str.contains(pattern, regex=True).astype(int)
    return df

def compute_pos_weights(df, aspects):
    weights = []
    for aspect in aspects:
        n_pos = max(df[aspect].sum(), 1)
        n_neg = len(df) - n_pos
        weights.append(n_neg / n_pos)
    return torch.tensor(weights, dtype=torch.float)

class AspectClassifier:
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
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(base)
        base_model = DistilBertForSequenceClassification.from_pretrained(
            base, num_labels=self.num_aspects,
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
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(load_dir)
        base_model = DistilBertForSequenceClassification.from_pretrained(
            base, num_labels=self.num_aspects,
            problem_type="multi_label_classification",
        )
        self.model = PeftModel.from_pretrained(base_model, load_dir)
        self.model.to(self.device)
        self.logger.info("Loaded from %s", load_dir)

    def predict(self, texts, threshold=0.5):
        self.model.eval()
        max_len = self.config["data"]["max_text_length"]
        encodings = self.tokenizer(texts, max_length=max_len, padding=True,
                                   truncation=True, return_tensors="pt")
        encodings = {k: v.to(self.device) for k, v in encodings.items()}
        with torch.no_grad():
            outputs = self.model(**encodings)
        probs = torch.sigmoid(outputs.logits).cpu().numpy()
        return (probs >= threshold).astype(int).tolist()


class AspectTrainer:
    def __init__(self, classifier, pos_weights=None):
        self.clf = classifier
        self.config = classifier.config
        self.logger = get_logger(__name__, self.config)
        self.model_cfg = self.config["models"]["aspect"]
        self.device = classifier.device
        self.pos_weights = pos_weights
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

        if self.pos_weights is not None:
            pw = self.pos_weights.to(self.device)
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
            self.logger.info("Weighted BCE pos_weights: %s",
                [f"{w:.1f}" for w in self.pos_weights.tolist()])
        else:
            self.loss_fn = nn.BCEWithLogitsLoss()

        optimizer = AdamW([p for p in model.parameters() if p.requires_grad],
                          lr=lr, weight_decay=weight_decay)
        total_steps = len(train_loader) * epochs
        scheduler = OneCycleLR(optimizer, max_lr=lr, total_steps=total_steps, pct_start=0.1)

        history = {"train_loss": [], "val_loss": [], "val_f1_micro": []}
        best_f1 = 0.0
        patience_counter = 0

        self.logger.info("Training: epochs=%d lr=%s batch=%d steps/epoch=%d",
            epochs, lr, self.model_cfg["batch_size"], len(train_loader))

        for epoch in range(1, epochs + 1):
            self.logger.info("")
            self.logger.info("=" * 60)
            self.logger.info("EPOCH %d/%d  patience=%d/%d  best_f1=%.4f",
                epoch, epochs, patience_counter, patience, best_f1)
            self.logger.info("=" * 60)
            t0 = time.time()

            train_loss = self._train_epoch(model, train_loader, optimizer, scheduler, max_grad_norm, epoch, epochs)
            self.logger.info("Validation ...")
            val_loss, val_f1_micro, val_f1_macro = self._eval_epoch(model, val_loader)
            elapsed = time.time() - t0

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_f1_micro"].append(val_f1_micro)

            self.logger.info("Epoch %d/%d | train=%.4f val=%.4f f1_micro=%.4f f1_macro=%.4f | %.1fmin",
                epoch, epochs, train_loss, val_loss, val_f1_micro, val_f1_macro, elapsed/60)

            if val_f1_micro > best_f1:
                best_f1 = val_f1_micro
                patience_counter = 0
                self.clf.save(save_dir / "best")
                self.logger.info("*** New best saved (f1_micro=%.4f) ***", best_f1)
            else:
                patience_counter += 1
                self.logger.info("No improvement. Patience %d/%d", patience_counter, patience)
                if patience_counter >= patience:
                    self.logger.info("Early stopping at epoch %d.", epoch)
                    break

            self.clf.save(save_dir / f"checkpoint_epoch{epoch}")

        with open(save_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)
        self.logger.info("Done. Best val F1 micro: %.4f", best_f1)
        return history

    def _train_epoch(self, model, loader, optimizer, scheduler, max_grad_norm, epoch, total_epochs):
        model.train()
        total_loss = 0.0
        n = len(loader)
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [train]", unit="batch", ncols=110)
        for step, batch in enumerate(pbar, 1):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = self.loss_fn(outputs.logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            avg = total_loss / step
            lr_now = scheduler.get_last_lr()[0]
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "avg": f"{avg:.4f}", "lr": f"{lr_now:.2e}"})
            if step % 100 == 0 or step == n:
                self.logger.info("  [Ep%d | %d/%d] loss=%.4f avg=%.4f lr=%.2e",
                    epoch, step, n, loss.item(), avg, lr_now)
        return total_loss / n

    def _eval_epoch(self, model, loader):
        model.eval()
        total_loss = 0.0
        all_preds, all_labels = [], []
        pbar = tqdm(loader, desc="Val", unit="batch", ncols=110, leave=False)
        with torch.no_grad():
            for batch in pbar:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = self.loss_fn(outputs.logits, labels)
                total_loss += loss.item()
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
    pbar = tqdm(test_loader, desc="Test aspects", unit="batch", ncols=110)
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
    logger.info("Aspect F1 micro=%.4f macro=%.4f hamming=%.4f",
        metrics["f1_micro"], metrics["f1_macro"], metrics["hamming_loss"])
    return metrics
