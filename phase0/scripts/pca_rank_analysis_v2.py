"""
Phase 0: PCA Effective Rank Analysis v2 (correct model loading)
Uses actual BIIC source files on VPS for model definition.
Verified: loads 50/50 parameters (same as exp_m_v3 scripts).
"""
import os
import sys
import json
import random
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict, Counter
from transformers import GPT2Tokenizer
from datasets import load_dataset

sys.path.insert(0, "/data/biic")
sys.path.insert(0, "/data/biic/phase2")

from clifford_cl41 import GRADE_SLICES, N_BLADES
from token_to_ic import TokenToImmutableCore
from mutable_state import BIICLayer

CHECKPOINT_PATH = "/data/biic/biic_lm/checkpoints/final.pt"
OUTPUT_DIR = "/data/biic/results/pca_analysis/"
HF_CACHE_DIR = "/data/biic/hf_cache"
RANDOM_STATE = 42
device = torch.device("cpu")

CANDIDATE_WORDS = [
    "run", "bank", "light", "right", "plant", "state", "match", "current",
    "spring", "rock", "fair", "bear", "interest", "play", "set", "turn",
    "lead", "book", "train", "file", "fire", "mean", "note", "round",
    "key", "draw", "figure", "ground", "charge", "check", "clear", "deal",
    "drive", "drop", "fall", "feed", "feel", "field", "flat", "force",
    "form", "frame", "grant", "head", "hold", "kind", "land", "last",
    "lay", "leave", "level"
]

os.makedirs(OUTPUT_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(OUTPUT_DIR, "errors.log"), encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# === Model (matches checkpoint - verified 50/50 in exp_m_v3) ===

class PositionEncoding(nn.Module):
    def __init__(self, max_len=2048, n_channels=32):
        super().__init__()
        self.pos_embed = nn.Embedding(max_len, n_channels * 10)
        self.n_channels = n_channels

    def forward(self, mv):
        B, L, C, D = mv.shape
        pos_ids = torch.arange(L, device=mv.device)
        pos_vec = self.pos_embed(pos_ids)
        pos_vec = pos_vec.reshape(L, C, 10) * 0.1
        g2_start, g2_end = GRADE_SLICES[2]
        mv_out = mv.clone()
        mv_out[:, :, :, g2_start:g2_end] = mv[:, :, :, g2_start:g2_end] + pos_vec.unsqueeze(0)
        return mv_out


class SimpleTokenMixer(nn.Module):
    def __init__(self, n_channels, kernel_size=8):
        super().__init__()
        dim = n_channels * N_BLADES
        self.conv = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size-1, groups=dim)
        self.gate = nn.Linear(dim, dim)

    def forward(self, mv):
        B, L, C, D = mv.shape
        x = mv.reshape(B, L, C * D)
        x_t = x.transpose(1, 2)
        conv_out = self.conv(x_t)[:, :, :L]
        conv_out = conv_out.transpose(1, 2)
        gate = torch.sigmoid(self.gate(x))
        mixed = x + gate * conv_out
        return mixed.reshape(B, L, C, D)


class BIICLMBlock(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.mixer = SimpleTokenMixer(n_channels)
        self.biic_layer = BIICLayer(n_channels, use_residual=True, use_eraser=True)

    def forward(self, mv, mv_prior):
        mv = self.mixer(mv)
        mv = self.biic_layer(mv, mv_prior)
        return mv


class BIICLanguageModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        C = config["n_channels"]
        self.encoder = TokenToImmutableCore(config["vocab_size"], C)
        self.pos_enc = PositionEncoding(max_len=2048, n_channels=C)
        self.blocks = nn.ModuleList([BIICLMBlock(C) for _ in range(config["n_layers"])])
        for block in self.blocks:
            if hasattr(block.biic_layer, "mutable_layer") and hasattr(block.biic_layer.mutable_layer, "eraser"):
                with torch.no_grad():
                    block.biic_layer.mutable_layer.eraser.decay_logits.fill_(config.get("eraser_init", 0.0))

    def encode(self, token_ids):
        mv = self.encoder(token_ids)
        mv = self.pos_enc(mv)
        mv_prior = mv.clone().detach()
        for block in self.blocks:
            mv = block(mv, mv_prior)
        return mv

    def encode_embedding_only(self, token_ids):
        return self.encoder(token_ids)

    def encode_with_layer(self, token_ids, n_layers=3):
        mv = self.encoder(token_ids)
        mv = self.pos_enc(mv)
        mv_prior = mv.clone().detach()
        for i, block in enumerate(self.blocks):
            if i >= n_layers:
                break
            mv = block(mv, mv_prior)
        return mv


# === Data ===

def load_tokenizer():
    for p in ["/data/biic/gpt2_tokenizer", "/data/biic/models_cache/AI-ModelScope/gpt2"]:
        if os.path.exists(p):
            return GPT2Tokenizer.from_pretrained(p)
    raise RuntimeError("No tokenizer")


def load_corpus():
    logger.info("Loading WikiText-103 from cache...")
    try:
        ds = load_dataset("wikitext", "wikitext-103-v1", split="train", cache_dir=HF_CACHE_DIR)
        texts = [item["text"] for item in ds if item["text"].strip() and len(item["text"]) > 50]
        texts = texts[:20000]
        logger.info("Loaded %d texts" % len(texts))
        return texts
    except Exception as e:
        logger.warning("Train failed: %s, trying test" % e)
        ds = load_dataset("wikitext", "wikitext-103-v1", split="test", cache_dir=HF_CACHE_DIR)
        texts = [item["text"] for item in ds if item["text"].strip() and len(item["text"]) > 50]
        logger.info("Loaded %d texts (test)" % len(texts))
        return texts


# === Step 1: Build Target Words ===

def build_target_words(corpus_texts, tokenizer, min_count=30, max_samples=100):
    logger.info("Building target word list...")
    random.seed(RANDOM_STATE)

    word_token_map = {}
    for word in CANDIDATE_WORDS:
        ids = tokenizer.encode(" " + word, add_special_tokens=False)
        if len(ids) == 1:
            word_token_map[word] = ids[0]
    logger.info("Single-token candidates: %d/%d" % (len(word_token_map), len(CANDIDATE_WORDS)))

    word_occurrences = defaultdict(list)
    for text in corpus_texts:
        if not text.strip():
            continue
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) < 15:
            continue
        for start in range(0, len(tokens) - 14, 15):
            end = min(start + 50, len(tokens))
            sent = tokens[start:end]
            if len(sent) < 15:
                continue
            for pos in range(len(sent)):
                for word, wid in word_token_map.items():
                    if sent[pos] == wid and len(word_occurrences[word]) < 500:
                        word_occurrences[word].append({
                            "sentence_tokens": sent,
                            "position": pos,
                            "prev_2_tokens": sent[max(0, pos-2):pos],
                            "next_2_tokens": sent[pos+1:min(len(sent), pos+3)],
                            "next_token": sent[pos+1] if pos+1 < len(sent) else None,
                        })

    result = {}
    skipped = []
    for word in sorted(word_occurrences.keys(), key=lambda w: len(word_occurrences[w]), reverse=True):
        occs = word_occurrences[word]
        if len(occs) < min_count:
            skipped.append(word)
            continue
        if len(occs) > max_samples:
            random.shuffle(occs)
            occs = occs[:max_samples]
        result[word] = occs

    logger.info("Target words: %d (skipped %d)" % (len(result), len(skipped)))
    for word in list(result.keys())[:5]:
        logger.info("  %s: %d samples" % (word, len(result[word])))

    meta = {"words": list(result.keys()), "skipped": skipped,
            "sentences_per_word": {w: len(v) for w, v in result.items()}}
    with open(os.path.join(OUTPUT_DIR, "target_words.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return result


# === Step 2: Extract Representations ===

def extract_representations(model, tokenizer, word_sentences):
    logger.info("Extracting representations for %d words..." % len(word_sentences))
    model.eval()
    results = {}
    g0_s, g0_e = GRADE_SLICES[0]
    g2_s, g2_e = GRADE_SLICES[2]

    for wi, (word, occs) in enumerate(word_sentences.items()):
        logger.info("  [%d/%d] %s (%d)" % (wi+1, len(word_sentences), word, len(occs)))
        eg0, eg2, g0, g2, hf, h3, ctx = [], [], [], [], [], [], []

        for occ in occs:
            try:
                sent = occ["sentence_tokens"]
                pos = occ["position"]
                ids = torch.tensor([sent], device=device)
                with torch.no_grad():
                    mv_e = model.encode_embedding_only(ids)
                    mv_3 = model.encode_with_layer(ids, n_layers=3)
                    mv_f = model.encode(ids)
                me = mv_e[0, pos]
                m3 = mv_3[0, pos]
                mf = mv_f[0, pos]
                eg0.append(me[:, g0_s:g0_e].reshape(-1).numpy())
                eg2.append(me[:, g2_s:g2_e].reshape(-1).numpy())
                g0.append(mf[:, g0_s:g0_e].reshape(-1).numpy())
                g2.append(mf[:, g2_s:g2_e].reshape(-1).numpy())
                hf.append(mf.reshape(-1).numpy())
                h3.append(m3.reshape(-1).numpy())
                ctx.append({"prev_2_tokens": occ["prev_2_tokens"], "next_token": occ["next_token"]})
            except Exception as e:
                continue

        if len(g0) < 5:
            continue
        results[word] = {
            "embed_grade0": np.array(eg0, dtype=np.float32),
            "embed_grade2": np.array(eg2, dtype=np.float32),
            "grade0": np.array(g0, dtype=np.float32),
            "grade2": np.array(g2, dtype=np.float32),
            "hidden_final": np.array(hf, dtype=np.float32),
            "hidden_layer3": np.array(h3, dtype=np.float32),
            "contexts": ctx,
        }
        logger.info("    %d reps (g0=%s g2=%s)" % (len(g0), results[word]["grade0"].shape, results[word]["grade2"].shape))
    return results


# === Step 3: SVD Analysis ===

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


def analyze_single(R, contexts):
    n, d = R.shape
    R_c = R - R.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(R_c, full_matrices=False)
    var = S ** 2
    total = var.sum()
    if total < 1e-12:
        return None
    cumvar = np.cumsum(var) / total
    result = {
        "rank_90": int(np.searchsorted(cumvar, 0.90) + 1),
        "rank_99": int(np.searchsorted(cumvar, 0.99) + 1),
    }
    S4 = var ** 2
    PR = float(total ** 2 / S4.sum()) if S4.sum() > 0 else 0
    result["PR"] = PR

    rng = np.random.RandomState(42)
    R_sh = R_c.copy()
    for col in range(d):
        rng.shuffle(R_sh[:, col])
    _, Sr, _ = np.linalg.svd(R_sh, full_matrices=False)
    S2r, S4r = Sr**2, Sr**4
    PR_r = float((S2r.sum())**2 / S4r.sum()) if S4r.sum() > 0 else 1.0
    result["PR_random"] = PR_r
    result["snr_ratio"] = float(PR / PR_r) if PR_r > 0 else 0

    cos_mat = cosine_similarity(R_c)
    triu = np.triu_indices(n, k=1)
    result["cos_median"] = float(np.median(cos_mat[triu]))

    try:
        if n >= 10:
            km = KMeans(n_clusters=2, random_state=42, n_init=10)
            labels = km.fit_predict(R_c)
            cdist = float(np.linalg.norm(km.cluster_centers_[0] - km.cluster_centers_[1]))
            intra = [R_c[labels == c].std(axis=0).mean() for c in range(2) if (labels == c).sum() > 1]
            result["separation_ratio"] = float(cdist / (np.mean(intra) + 1e-8)) if intra else None
        else:
            result["separation_ratio"] = None
    except:
        result["separation_ratio"] = None

    try:
        next_toks = [c.get("next_token") for c in contexts]
        if len(next_toks) == n:
            counts = Counter([t for t in next_toks if t is not None])
            top2 = [t for t, _ in counts.most_common(2)]
            if len(top2) == 2:
                mask = np.array([t in top2 for t in next_toks])
                if mask.sum() >= 10:
                    X = R_c[mask]
                    y = np.array([0 if next_toks[i] == top2[0] else 1 for i in range(n) if mask[i]])
                    if len(np.unique(y)) == 2 and min(Counter(y.tolist()).values()) >= 3:
                        clf = LogisticRegression(max_iter=200, random_state=42)
                        cv = min(3, min(Counter(y.tolist()).values()))
                        scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy") if cv >= 2 else [0.5]
                        result["probe_accuracy"] = float(np.mean(scores))
                    else:
                        result["probe_accuracy"] = None
                else:
                    result["probe_accuracy"] = None
            else:
                result["probe_accuracy"] = None
        else:
            result["probe_accuracy"] = None
    except:
        result["probe_accuracy"] = None
    return result


def analyze_representations(representations):
    rep_types = ["embed_grade0", "embed_grade2", "grade0", "grade2", "hidden_layer3", "hidden_final"]
    per_word = {}
    for word, data in representations.items():
        per_word[word] = {}
        for rt in rep_types:
            if rt not in data:
                continue
            R = data[rt]
            if R.shape[0] < 5:
                continue
            try:
                r = analyze_single(R, data.get("contexts", []))
                if r:
                    per_word[word][rt] = r
            except Exception as e:
                logger.warning("Fail %s/%s: %s" % (word, rt, e))
    return per_word


# === Step 4: Summary ===

def generate_summary(per_word_results):
    rep_types = ["embed_grade0", "embed_grade2", "grade0", "grade2", "hidden_layer3", "hidden_final"]
    summary = {}
    for rt in rep_types:
        vals = defaultdict(list)
        for word, wr in per_word_results.items():
            if rt in wr:
                for m in ["PR", "rank_90", "cos_median", "snr_ratio"]:
                    if m in wr[rt] and wr[rt][m] is not None:
                        vals[m].append(wr[rt][m])
        summary[rt] = {}
        for m, v in vals.items():
            if v:
                arr = np.array(v)
                summary[rt]["%s_median" % m] = float(np.median(arr))
                summary[rt]["%s_p95" % m] = float(np.percentile(arr, 95))

    g2_pr = summary.get("grade2", {}).get("PR_p95")
    if g2_pr is None:
        decision = "insufficient data"
    elif g2_pr <= 64:
        decision = "SFE k=64 feasible, linear assumption holds"
    elif g2_pr <= 128:
        decision = "SFE needs k=128, linear assumption largely holds"
    elif g2_pr <= 256:
        decision = "SFE-v2 recommended: sparse activation over k=256 basis"
    else:
        decision = "linear assumption questionable, need nonlinear component"
    summary["decision"] = decision
    summary["grade2_PR_p95"] = g2_pr
    return summary


# === Main ===

def main():
    logger.info("=" * 60)
    logger.info("Phase 0: PCA Effective Rank Analysis v2")
    logger.info("=" * 60)

    tokenizer = load_tokenizer()
    corpus = load_corpus()
    word_sentences = build_target_words(corpus, tokenizer, min_count=30, max_samples=100)
    if not word_sentences:
        logger.error("No target words!")
        return

    logger.info("Loading model...")
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    model = BIICLanguageModel(config)
    state = ckpt["model_state"]
    ms = model.state_dict()
    loaded = sum(1 for k in ms if k in state and ms[k].shape == state[k].shape)
    for k in ms:
        if k in state and ms[k].shape == state[k].shape:
            ms[k] = state[k]
    model.load_state_dict(ms)
    model.eval()
    logger.info("Model: %d/%d params loaded" % (loaded, len(ms)))

    representations = extract_representations(model, tokenizer, word_sentences)
    if not representations:
        logger.error("No representations!")
        return

    logger.info("Step 3: Analysis...")
    per_word = analyze_representations(representations)
    summary = generate_summary(per_word)

    results = {"summary": summary, "per_word": per_word}
    results["summary"]["n_words"] = len(per_word)
    results["summary"]["params_loaded"] = "%d/%d" % (loaded, len(ms))

    with open(os.path.join(OUTPUT_DIR, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("Saved results.json")

    print("\n" + "=" * 60)
    print("PCA RANK ANALYSIS SUMMARY")
    print("=" * 60)
    print("Words: %d | Params: %d/%d" % (len(per_word), loaded, len(ms)))
    for rt in ["embed_grade0", "embed_grade2", "grade0", "grade2", "hidden_layer3", "hidden_final"]:
        s = summary.get(rt, {})
        if s.get("PR_median") is not None:
            print("  [%s] PR=%.1f/%.1f rank90=%.0f/%.0f cos=%.3f snr=%.2f" % (
                rt, s["PR_median"], s["PR_p95"],
                s.get("rank_90_median", 0), s.get("rank_90_p95", 0),
                s.get("cos_median_median", 0), s.get("snr_ratio_median", 0)))
    print("\n  grade2 PR p95: %s" % summary.get("grade2_PR_p95", "N/A"))
    print("  DECISION: %s" % summary["decision"])
    print("=" * 60)


if __name__ == "__main__":
    main()
