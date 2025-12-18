import torch
import pandas as pd
import numpy as np
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    GPT2LMHeadModel,
    BertTokenizerFast,
)
from bert_score import score
import math


class Evaluator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Initializing Evaluators on {self.device}...")

        # 1. Load Style Judge (The one you are training right now)
        self.judge_tokenizer = BertTokenizer.from_pretrained("models/bert_judge_zh")
        self.judge_model = BertForSequenceClassification.from_pretrained(
            "models/bert_judge_zh"
        ).to(self.device)
        self.judge_model.eval()

        # 2. Load Fluency Judge (GPT2-Chinese)
        # We use a standard pre-trained Chinese GPT2
        self.gpt_tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")
        self.gpt_model = GPT2LMHeadModel.from_pretrained(
            "ckiplab/gpt2-base-chinese"
        ).to(self.device)
        self.gpt_model.eval()

    def get_style_accuracy(self, texts):
        """Returns % of texts classified as Neutral (Label 0)"""
        inputs = self.judge_tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=128
        ).to(self.device)
        with torch.no_grad():
            logits = self.judge_model(**inputs).logits
            preds = torch.argmax(logits, dim=1)

        # Label 0 = Neutral. We want to know how many are 0.
        neutral_count = (preds == 0).sum().item()
        return neutral_count / len(texts)

    def get_perplexity(self, texts):
        """Calculates PPL (Fluency)"""
        # Simplified PPL calculation
        ppls = []
        for text in texts:
            encodings = self.gpt_tokenizer(text, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.gpt_model(**encodings, labels=encodings.input_ids)
                loss = outputs.loss
                ppl = math.exp(loss.item())
                if not math.isnan(ppl):
                    ppls.append(ppl)
        return np.mean(ppls) if ppls else float("inf")

    def get_bert_score(self, candidates, references):
        """Calculates Content Preservation"""
        # candidates = your model output
        # references = the original input (source) or the gold target
        P, R, F1 = score(
            candidates, references, lang="zh", verbose=False, device=self.device
        )
        return F1.mean().item()


# --- USAGE EXAMPLE ---
if __name__ == "__main__":
    # 1. Load your GOLD Data (From friends)
    # If you don't have it yet, use a small slice of Synthetic Val for testing
    try:
        df = pd.read_csv("data/processed/test_chinese_gold.csv")
        sources = df["Chinese_Biased"].tolist()
        golds = df["Chinese_Neutral_Gold"].tolist()
    except:
        print("Gold data not found. Using dummy data for testing.")
        sources = ["这个激进的政权失败了。", "他愚蠢地拒绝了。"]
        golds = ["这个政府失败了。", "他拒绝了。"]

    # 2. Load YOUR Generator (mBART Model 3)
    # ... (Load code here, generate outputs) ...
    # For now, let's assume you have a list of outputs
    generated_outputs = ["这个政府失败了。", "他拒绝了。"]  # Dummy outputs

    # 3. Run Evaluation
    evaluator = Evaluator()

    print("\n--- METRICS REPORT ---")

    # Metric 1: Accuracy (Higher is better)
    acc = evaluator.get_style_accuracy(generated_outputs)
    print(f"Style Transfer Accuracy: {acc*100:.2f}%")

    # Metric 2: Content Preservation (Higher is better)
    # We compare Output vs Input to ensure we didn't change facts
    content_score = evaluator.get_bert_score(generated_outputs, sources)
    print(f"Content Preservation (BERTScore): {content_score:.4f}")

    # Metric 3: Fluency (Lower is better)
    ppl = evaluator.get_perplexity(generated_outputs)
    print(f"Fluency (Perplexity): {ppl:.2f}")
