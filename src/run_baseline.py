import os
import json
from typing import List

import torch
from torch.utils.data import DataLoader

from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model
import sacrebleu
from tqdm.auto import tqdm


# ----------------- Config -----------------
DATA_DIR = "../data/data-simplification/wikilarge"
MODEL_NAME = "t5-base"

MAX_SOURCE_LENGTH = 128
MAX_TARGET_LENGTH = 128

NUM_EPOCHS = 1
TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 32
LR = 5e-5

MAX_TRAIN_SAMPLES = None
TRAIN_TYPE = "none"  # fine-tuning/lora/none


# ---------- Simple SARI implementation (no easse) ----------

def _get_ngrams(tokens: List[str], n: int):
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def _sari_sentence(src: str, cand: str, refs: List[str], max_n: int = 4) -> float:
    src_toks = src.split()
    cand_toks = cand.split()
    refs_toks = [r.split() for r in refs]

    F_add_total = F_keep_total = F_del_total = 0.0
    n_count = 0

    for n in range(1, max_n+1):
        n_count += 1
        src_ngrams = set(_get_ngrams(src_toks, n))
        cand_ngrams = set(_get_ngrams(cand_toks, n))
        refs_ngrams_list = [set(_get_ngrams(rt, n)) for rt in refs_toks]

        # Added
        cand_add = cand_ngrams - src_ngrams
        refs_add = set().union(*[(r - src_ngrams) for r in refs_ngrams_list]) if refs_ngrams_list else set()
        overlap_add = cand_add & refs_add
        P_add = len(overlap_add) / len(cand_add) if cand_add else 0.0
        R_add = len(overlap_add) / len(refs_add) if refs_add else 0.0
        F_add = 2*P_add*R_add/(P_add+R_add) if (P_add+R_add) > 0 else 0.0

        # Kept
        cand_keep = cand_ngrams & src_ngrams
        refs_keep = set().union(*[(r & src_ngrams) for r in refs_ngrams_list]) if refs_ngrams_list else set()
        overlap_keep = cand_keep & refs_keep
        P_keep = len(overlap_keep) / len(cand_keep) if cand_keep else 0.0
        R_keep = len(overlap_keep) / len(refs_keep) if refs_keep else 0.0
        F_keep = 2*P_keep*R_keep/(P_keep+R_keep) if (P_keep+R_keep) > 0 else 0.0

        # Deleted
        src_del = src_ngrams - cand_ngrams
        refs_del = set().union(*[(src_ngrams - r) for r in refs_ngrams_list]) if refs_ngrams_list else set()
        overlap_del = src_del & refs_del
        P_del = len(overlap_del) / len(src_del) if src_del else 0.0
        R_del = len(overlap_del) / len(refs_del) if refs_del else 0.0
        F_del = 2*P_del*R_del/(P_del+R_del) if (P_del+R_del) > 0 else 0.0

        F_add_total += F_add
        F_keep_total += F_keep
        F_del_total += F_del

    F_add_avg = F_add_total / n_count
    F_keep_avg = F_keep_total / n_count
    F_del_avg = F_del_total / n_count

    return (F_add_avg + F_keep_avg + F_del_avg) / 3.0

def sari_corpus(sources: List[str], candidates: List[str], references: List[List[str]]) -> float:
    scores = [
        _sari_sentence(s, c, rs)
        for s, c, rs in zip(sources, candidates, references)
    ]
    return sum(scores) / len(scores)


# ---------- Load WikiLarge ----------

def read_parallel(src_path, dst_path, filter_short=True, filter_ratio=True):
    with open(src_path, "r", encoding="utf-8") as f_src, \
         open(dst_path, "r", encoding="utf-8") as f_dst:
        src_lines = [line.strip() for line in f_src.readlines()]
        dst_lines = [line.strip() for line in f_dst.readlines()]

    assert len(src_lines) == len(dst_lines)
    src_out, dst_out = [], []

    for s, d in zip(src_lines, dst_lines):
        if not s or not d:
            continue
        if filter_short and (len(s.split()) < 5 or len(d.split()) < 3):
            continue
        if filter_ratio and len(d.split()) >= len(s.split()):
            continue
        ratio = len(d.split()) / len(s.split())
        if filter_ratio and ratio >= 1:
            continue
        src_out.append(s)
        dst_out.append(d)

    return {"source": src_out, "target": dst_out}

def load_wikilarge(data_dir):
    train = read_parallel(
        os.path.join(data_dir, "wiki.full.aner.train.src"),
        os.path.join(data_dir, "wiki.full.aner.train.dst"),
        filter_short=True,
        filter_ratio=True
    )
    valid = read_parallel(
        os.path.join(data_dir, "wiki.full.aner.valid.src"),
        os.path.join(data_dir, "wiki.full.aner.valid.dst"),
        filter_short=True,
        filter_ratio=True
    )
    test = read_parallel(
        os.path.join(data_dir, "wiki.full.aner.test.src"),
        os.path.join(data_dir, "wiki.full.aner.test.dst"),
        filter_short=False,
        filter_ratio=False
    )

    ds_train = Dataset.from_dict(train)
    ds_valid = Dataset.from_dict(valid)
    ds_test = Dataset.from_dict(test)

    return DatasetDict(train=ds_train, validation=ds_valid, test=ds_test)
  

# ---------- Tokenizer + model ----------

def preprocess(examples, tokenizer):
    model_inputs = tokenizer(
        examples["source"],
        max_length=MAX_SOURCE_LENGTH,
        truncation=True,
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["target"],
            max_length=MAX_TARGET_LENGTH,
            truncation=True,
        )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# ---------- Manual training loop (fine-tuning) ----------

def fine_tuning(model, tokenizer, tokenized_datasets, data_collator):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    model.to(device)

    train_dataset_tok = tokenized_datasets["train"]
    if MAX_TRAIN_SAMPLES is not None and MAX_TRAIN_SAMPLES < len(train_dataset_tok):
        train_dataset_tok = train_dataset_tok.select(range(MAX_TRAIN_SAMPLES))
    
    train_loader = DataLoader(
        train_dataset_tok,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        collate_fn=data_collator,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    print("Start training on subset:", len(train_dataset_tok), "examples")
    model.train()
    for epoch in range(NUM_EPOCHS):
        epoch_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)
        for batch in epoch_bar:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_bar.set_postfix({"loss": loss.item()})
    
    return model


# ---------- Manual training loop (lora) ----------

def lora(model, tokenzier, tokenized_datasets, data_collator):
    # --- LoRA config ---
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q", "k", "v", "o"],  # T5 attention layers
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_2_SEQ_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    model.to(device)

    # ---------- Data ----------
    train_dataset_tok = tokenized_datasets["train"]
    if MAX_TRAIN_SAMPLES is not None and MAX_TRAIN_SAMPLES < len(train_dataset_tok):
        train_dataset_tok = train_dataset_tok.select(range(MAX_TRAIN_SAMPLES))

    train_loader = DataLoader(
        train_dataset_tok,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        collate_fn=data_collator,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # ---------- Training loop ----------
    print("Start training on subset:", len(train_dataset_tok), "examples")
    model.train()
    for epoch in range(NUM_EPOCHS):
        epoch_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)
        for batch in epoch_bar:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_bar.set_postfix({"loss": loss.item()})
    
    return model


# ---------- Evaluation: BLEU + SARI (with tqdm) ----------

def evaluate(model, tokenizer, raw_datasets, tokenized_datasets):
    print("Evaluating on test set (BLEU + SARI)...")
    test_dataset_tok = tokenized_datasets["test"]
    test_raw = raw_datasets["test"]

    test_loader = DataLoader(
        test_dataset_tok,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        collate_fn=data_collator,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    model.to(device)

    pred_texts = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Generating", leave=False):
            batch = {
                k: v.to(device)
                for k, v in batch.items()
                if k in ["input_ids", "attention_mask"]
            }
            outputs = model.generate(
                **batch,
                max_length=MAX_TARGET_LENGTH,
                num_beams=4,
            )
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            pred_texts.extend(decoded)

    # Align lengths
    test_src = test_raw["source"][: len(pred_texts)]
    test_ref = test_raw["target"][: len(pred_texts)]

    # BLEU (0–100)
    bleu = sacrebleu.corpus_bleu(pred_texts, [test_ref]).score

    # SARI (our implementation is 0–1; convert to 0–100 for reporting)
    sari_raw = sari_corpus(
        test_src,
        pred_texts,
        [[r] for r in test_ref],
    )
    sari = sari_raw * 100.0

    print(f"Test BLEU: {bleu:.2f}")
    print(f"Test SARI: {sari:.2f}")

    outputs = {
      "test_src": test_src,
      "test_ref": test_ref,
      "pred_texts": pred_texts
    }

    return bleu, sari, outputs


# ---------- Save result ----------

def save_result(model, tokenizer, bleu, sari, outputs, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    test_src = outputs["test_src"]
    test_ref = outputs["test_ref"]
    pred_texts = outputs["pred_texts"]

    # 1) Save model + tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # 2) Save metrics
    metrics = {"bleu": float(bleu), "sari": float(sari)}
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # 3) Save predictions for error analysis
    pred_path = os.path.join(output_dir, "predictions.tsv")
    with open(pred_path, "w", encoding="utf-8") as f:
        f.write("src\tref\tpred\n")
        for s, r, p in zip(test_src, test_ref, pred_texts):
            s_clean = s.replace("\t", " ")
            r_clean = r.replace("\t", " ")
            p_clean = p.replace("\t", " ")
            f.write(f"{s_clean}\t{r_clean}\t{p_clean}\n")

    print("Saved model and outputs to:", output_dir)


# ---------- Main ----------

if __name__ == "__main__":
    # Prompt for train type
    print("0: no training (by default)")
    print("1: fine-tuning")
    print("2: lora")
    input_type = int(input("Please input training type: "))
    if input_type == 1:
        TRAIN_TYPE = "fine-tuning"
    elif input_type == 2:
        TRAIN_TYPE = "lora"
    
    # Prompt for model type
    print("0: T5-Base (by default)")
    print("1: T5-small")
    input_type = int(input("Please input model type: "))
    if input_type == 1:
        MODEL_NAME = "t5-small"
  
    # Load dataset
    raw_datasets = load_wikilarge(DATA_DIR)

    # Load model and tokenizer
    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # Process dataset
    tokenized_datasets = raw_datasets.map(
        preprocess,
        fn_kwargs={"tokenizer": tokenizer},
        batched=True,
        remove_columns=["source", "target"],
    )
    print(tokenized_datasets)

    # Train model for type
    if TRAIN_TYPE == "none":
        print("No training needed!")
    elif TRAIN_TYPE == "fine-tuning":
        print("Start fine-tuning for all paramters")
        model = fine_tuning(model, tokenizer, tokenized_datasets, data_collator)
    elif TRAIN_TYPE == "lora":
        print("Start lora")
        model = lora(model, tokenizer, tokenized_datasets, data_collator)
    else:
        print("How did you get here?")
    
    # Evaluate model performance
    bleu, sari, outputs = evaluate(model, tokenizer, raw_datasets, tokenized_datasets)

    # Save model results
    output_names = [MODEL_NAME, TRAIN_TYPE, "baseline"]
    output_dir = "../results/" + "_".join(output_names).replace("-", "_")
    save_result(model, tokenizer, bleu, sari, outputs, output_dir)
