"""
Experiment M v3: Grade-2 Cosine Similarity Analysis
Instead of geometric product, test if dependency relations are encoded
in the DIRECTION of grade-2 vectors (cosine similarity).

Hypothesis: head-dependent pairs have systematically different cosine
similarity patterns than random pairs, and different dep types
(nsubj/obj/amod) have distinct cosine profiles.

Uses trained BIIC LM checkpoint.
Saves: /data/biic/results/exp_m_v3_cosine.json
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import json
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from collections import defaultdict
import random

sys.path.insert(0, "/data/biic")
sys.path.insert(0, "/data/biic/phase2")

from clifford_cl41 import GRADE_SLICES, N_BLADES
from token_to_ic import TokenToImmutableCore
from mutable_state import BIICLayer
from eraser_ops import GradeAwareEraser

device = torch.device("cpu")


# ============================================================
# Model Definition (same as exp_m_v2)
# ============================================================

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
            if hasattr(block.biic_layer.mutable_layer, "eraser"):
                with torch.no_grad():
                    block.biic_layer.mutable_layer.eraser.decay_logits.fill_(config.get("eraser_init", 0.0))

    def encode(self, token_ids):
        mv = self.encoder(token_ids)
        mv = self.pos_enc(mv)
        mv_prior = mv.clone().detach()
        for block in self.blocks:
            mv = block(mv, mv_prior)
        return mv


# ============================================================
# UD Data Parsing (same as exp_m_v2)
# ============================================================

def parse_conllu(path, max_sentences=2000):
    sentences = []
    cur_tokens = []
    cur_deps = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if cur_tokens:
                    sentences.append({"tokens": cur_tokens, "deps": cur_deps})
                    cur_tokens = []
                    cur_deps = []
                    if len(sentences) >= max_sentences:
                        break
                continue
            if line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 8:
                continue
            if "-" in parts[0] or "." in parts[0]:
                continue
            token = parts[1]
            head = int(parts[6]) if parts[6] != "_" else 0
            deprel = parts[7].split(":")[0] if parts[7] != "_" else "dep"
            cur_tokens.append(token)
            cur_deps.append({"head": head, "deprel": deprel, "idx": int(parts[0])})

    if cur_tokens:
        sentences.append({"tokens": cur_tokens, "deps": cur_deps})
    return sentences


def build_tokenizer():
    from transformers import GPT2Tokenizer
    tok_path = "/data/biic/gpt2_tokenizer"
    if os.path.exists(tok_path):
        return GPT2Tokenizer.from_pretrained(tok_path)
    cache_path = "/data/biic/models_cache/AI-ModelScope/gpt2"
    if os.path.exists(cache_path):
        return GPT2Tokenizer.from_pretrained(cache_path)
    raise RuntimeError("No GPT-2 tokenizer found")


def extract_dep_pairs(sentences, target_rels=("nsubj", "obj", "amod")):
    dep_pairs = []
    nodep_pairs = []

    for s_idx, sent in enumerate(sentences):
        n_tokens = len(sent["tokens"])
        if n_tokens < 3:
            continue

        dep_positions = set()
        for dep in sent["deps"]:
            if dep["deprel"] in target_rels and dep["head"] > 0:
                child_pos = dep["idx"] - 1
                head_pos = dep["head"] - 1
                if head_pos < n_tokens and child_pos < n_tokens:
                    dep_pairs.append((s_idx, head_pos, child_pos, dep["deprel"]))
                    dep_positions.add((head_pos, child_pos))

        # Non-dependency pairs from same sentence
        if n_tokens >= 4:
            rng = random.Random(s_idx)
            attempts = 0
            added = 0
            while added < 3 and attempts < 20:
                i = rng.randint(0, n_tokens - 1)
                j = rng.randint(0, n_tokens - 1)
                if i != j and (i, j) not in dep_positions and (j, i) not in dep_positions:
                    has_dep = False
                    for dep in sent["deps"]:
                        ci = dep["idx"] - 1
                        hi = dep["head"] - 1
                        if (ci == i and hi == j) or (ci == j and hi == i):
                            has_dep = True
                            break
                    if not has_dep:
                        nodep_pairs.append((s_idx, i, j, "none"))
                        added += 1
                attempts += 1

    return dep_pairs, nodep_pairs


# ============================================================
# Main Experiment
# ============================================================

def run_experiment():
    print("=" * 60)
    print("Experiment M v3: Grade-2 Cosine Similarity Analysis")
    print("=" * 60)

    # Load checkpoint
    ckpt_path = "/data/biic/biic_lm/checkpoints/final.pt"
    print("Loading checkpoint:", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    print("Config: n_channels=%d, n_layers=%d, step=%d" % (config["n_channels"], config["n_layers"], ckpt["step"]))

    model = BIICLanguageModel(config)
    state = ckpt["model_state"]
    model_state = model.state_dict()
    loaded = 0
    for k in model_state:
        if k in state:
            model_state[k] = state[k]
            loaded += 1
    model.load_state_dict(model_state)
    model.eval()
    print("Loaded %d/%d parameters" % (loaded, len(model_state)))

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = build_tokenizer()

    # Parse UD data
    print("\nParsing UD English EWT...")
    ud_path = "/data/biic/ud_english_ewt/en_ewt-ud-train.conllu"
    sentences = parse_conllu(ud_path, max_sentences=1500)
    print("Loaded %d sentences" % len(sentences))

    # Extract dependency pairs
    print("\nExtracting dependency pairs...")
    dep_pairs, nodep_pairs = extract_dep_pairs(sentences)
    print("  nsubj/obj/amod pairs: %d" % len(dep_pairs))
    print("  non-dep pairs: %d" % len(nodep_pairs))

    # Encode sentences
    print("\nEncoding sentences...")
    encoded_cache = {}
    g2_start, g2_end = GRADE_SLICES[2]

    def get_encoded_sentence(s_idx):
        if s_idx not in encoded_cache:
            tokens = sentences[s_idx]["tokens"]
            text = " ".join(tokens)
            input_ids = tokenizer.encode(text, add_special_tokens=False)
            word_to_token = []
            cur_pos = 0
            for wi, word in enumerate(tokens):
                word_tokens = tokenizer.encode(" " + word if wi > 0 else word, add_special_tokens=False)
                word_to_token.append(cur_pos)
                cur_pos += len(word_tokens)
            with torch.no_grad():
                ids_tensor = torch.tensor([input_ids], device=device)
                if ids_tensor.shape[1] > 128:
                    ids_tensor = ids_tensor[:, :128]
                mv = model.encode(ids_tensor)  # [1, L, C, 32]
            encoded_cache[s_idx] = {"mv": mv[0], "word_to_token": word_to_token, "n_tokens": mv.shape[1]}
        return encoded_cache[s_idx]

    def get_grade2_flat(enc, word_pos):
        """Get flattened grade-2 vector [C*10] for a word position"""
        wtt = enc["word_to_token"]
        if word_pos >= len(wtt):
            return None
        tok_pos = wtt[word_pos]
        if tok_pos >= enc["n_tokens"]:
            return None
        mv_tok = enc["mv"][tok_pos]  # [C, 32]
        g2 = mv_tok[:, g2_start:g2_end]  # [C, 10]
        return g2.reshape(-1)  # [C*10]

    # ============================================================
    # Test 1: Cosine similarity between head-dep pairs
    # ============================================================
    print("\n" + "=" * 60)
    print("Test 1: Cosine similarity (grade-2 direction)")
    print("=" * 60)

    cosines_by_rel = defaultdict(list)
    diff_norms_by_rel = defaultdict(list)
    features_all = []
    labels_all = []

    print("  Processing dep pairs...")
    for s_idx, head_pos, child_pos, rel in dep_pairs:
        try:
            enc = get_encoded_sentence(s_idx)
            g2_head = get_grade2_flat(enc, head_pos)
            g2_child = get_grade2_flat(enc, child_pos)
            if g2_head is None or g2_child is None:
                continue

            # Cosine similarity
            cos = F.cosine_similarity(g2_head.unsqueeze(0), g2_child.unsqueeze(0)).item()
            cosines_by_rel[rel].append(cos)

            # Difference norm
            diff_norm = (g2_head - g2_child).norm().item()
            diff_norms_by_rel[rel].append(diff_norm)

            # Features for probe: [cosine, diff_norm, head_norm, child_norm, dot_product]
            feat = np.array([
                cos,
                diff_norm,
                g2_head.norm().item(),
                g2_child.norm().item(),
                torch.dot(g2_head, g2_child).item(),
                # Per-channel cosines (top 8 channels)
            ])
            # Add per-channel cosine similarities
            C = config["n_channels"]
            g2_h_ch = g2_head.reshape(C, 10)
            g2_c_ch = g2_child.reshape(C, 10)
            ch_cosines = F.cosine_similarity(g2_h_ch, g2_c_ch, dim=1).numpy()
            feat = np.concatenate([feat, ch_cosines])

            features_all.append(feat)
            labels_all.append(rel)
        except Exception:
            continue

    print("  Processing non-dep pairs...")
    for s_idx, i, j, _ in nodep_pairs:
        try:
            enc = get_encoded_sentence(s_idx)
            g2_i = get_grade2_flat(enc, i)
            g2_j = get_grade2_flat(enc, j)
            if g2_i is None or g2_j is None:
                continue

            cos = F.cosine_similarity(g2_i.unsqueeze(0), g2_j.unsqueeze(0)).item()
            cosines_by_rel["none"].append(cos)

            diff_norm = (g2_i - g2_j).norm().item()
            diff_norms_by_rel["none"].append(diff_norm)

            feat = np.array([
                cos,
                diff_norm,
                g2_i.norm().item(),
                g2_j.norm().item(),
                torch.dot(g2_i, g2_j).item(),
            ])
            C = config["n_channels"]
            g2_i_ch = g2_i.reshape(C, 10)
            g2_j_ch = g2_j.reshape(C, 10)
            ch_cosines = F.cosine_similarity(g2_i_ch, g2_j_ch, dim=1).numpy()
            feat = np.concatenate([feat, ch_cosines])

            features_all.append(feat)
            labels_all.append("none")
        except Exception:
            continue

    # ============================================================
    # Analysis
    # ============================================================
    print("\n" + "=" * 60)
    print("Results: Cosine Similarity by Relation Type")
    print("=" * 60)

    results = {"cosine_by_rel": {}, "diff_norm_by_rel": {}}

    for rel in ["nsubj", "obj", "amod", "none"]:
        cos_arr = np.array(cosines_by_rel[rel])
        diff_arr = np.array(diff_norms_by_rel[rel])
        if len(cos_arr) > 0:
            results["cosine_by_rel"][rel] = {
                "mean": float(cos_arr.mean()),
                "std": float(cos_arr.std()),
                "n": len(cos_arr),
            }
            results["diff_norm_by_rel"][rel] = {
                "mean": float(diff_arr.mean()),
                "std": float(diff_arr.std()),
                "n": len(diff_arr),
            }
            print("  %s: cosine=%.4f+/-%.4f, diff_norm=%.4f+/-%.4f (n=%d)" % (
                rel, cos_arr.mean(), cos_arr.std(), diff_arr.mean(), diff_arr.std(), len(cos_arr)))

    # t-tests: each dep type vs none
    print("\n" + "=" * 60)
    print("t-tests: dep type vs non-dep (cosine)")
    print("=" * 60)

    ttest_results = {}
    none_cos = np.array(cosines_by_rel["none"])
    for rel in ["nsubj", "obj", "amod"]:
        rel_cos = np.array(cosines_by_rel[rel])
        if len(rel_cos) > 10 and len(none_cos) > 10:
            t_stat, p_val = stats.ttest_ind(rel_cos, none_cos)
            pooled_std = np.sqrt((rel_cos.std()**2 + none_cos.std()**2) / 2)
            d = (rel_cos.mean() - none_cos.mean()) / pooled_std if pooled_std > 0 else 0
            ttest_results[rel] = {
                "t_stat": float(t_stat),
                "p_value": float(p_val),
                "cohens_d": float(d),
                "significant": bool(p_val < 0.05),
            }
            print("  %s vs none: t=%.3f, p=%.6f, d=%.4f %s" % (
                rel, t_stat, p_val, d, "*" if p_val < 0.05 else ""))

    # Combined: all dep vs none
    all_dep_cos = np.concatenate([np.array(cosines_by_rel[r]) for r in ["nsubj", "obj", "amod"]])
    t_stat_all, p_val_all = stats.ttest_ind(all_dep_cos, none_cos)
    pooled_std_all = np.sqrt((all_dep_cos.std()**2 + none_cos.std()**2) / 2)
    d_all = (all_dep_cos.mean() - none_cos.mean()) / pooled_std_all if pooled_std_all > 0 else 0
    print("  ALL_DEP vs none: t=%.3f, p=%.6f, d=%.4f %s" % (t_stat_all, p_val_all, d_all, "*" if p_val_all < 0.05 else ""))

    results["ttest_all_dep_vs_none"] = {
        "t_stat": float(t_stat_all),
        "p_value": float(p_val_all),
        "cohens_d": float(d_all),
    }

    # t-tests on diff norms
    print("\n" + "=" * 60)
    print("t-tests: dep type vs non-dep (diff norm)")
    print("=" * 60)

    ttest_diff_results = {}
    none_diff = np.array(diff_norms_by_rel["none"])
    for rel in ["nsubj", "obj", "amod"]:
        rel_diff = np.array(diff_norms_by_rel[rel])
        if len(rel_diff) > 10 and len(none_diff) > 10:
            t_stat, p_val = stats.ttest_ind(rel_diff, none_diff)
            pooled_std = np.sqrt((rel_diff.std()**2 + none_diff.std()**2) / 2)
            d = (rel_diff.mean() - none_diff.mean()) / pooled_std if pooled_std > 0 else 0
            ttest_diff_results[rel] = {
                "t_stat": float(t_stat),
                "p_value": float(p_val),
                "cohens_d": float(d),
                "significant": bool(p_val < 0.05),
            }
            print("  %s vs none: t=%.3f, p=%.6f, d=%.4f %s" % (
                rel, t_stat, p_val, d, "*" if p_val < 0.05 else ""))

    # ============================================================
    # Test 2: Linear probe with richer features
    # ============================================================
    print("\n" + "=" * 60)
    print("Test 2: Linear probe (cosine + per-channel features)")
    print("=" * 60)

    X_all = np.array(features_all)
    y_all = np.array(labels_all)

    # 4-class classification
    label_map = {"nsubj": 0, "obj": 1, "amod": 2, "none": 3}
    y_encoded = np.array([label_map.get(l, 3) for l in y_all])

    clf = LogisticRegression(max_iter=2000, random_state=42)
    scores = cross_val_score(clf, X_all, y_encoded, cv=5, scoring="accuracy")
    probe_accuracy_4class = scores.mean()
    print("  4-class (nsubj/obj/amod/none) accuracy: %.4f (+/- %.4f)" % (probe_accuracy_4class, scores.std()))

    # Binary: dep vs non-dep
    y_binary = np.array([0 if l == "none" else 1 for l in y_all])
    scores_bin = cross_val_score(clf, X_all, y_binary, cv=5, scoring="accuracy")
    probe_accuracy_binary = scores_bin.mean()
    print("  Binary (dep vs non-dep) accuracy: %.4f (+/- %.4f)" % (probe_accuracy_binary, scores_bin.mean()))

    # Per-class accuracy
    clf.fit(X_all, y_encoded)
    dep_type_acc = {}
    for rel, idx in label_map.items():
        mask = y_encoded == idx
        if mask.sum() > 0:
            pred = clf.predict(X_all[mask])
            acc = (pred == idx).mean()
            dep_type_acc[rel] = float(acc)
            print("    %s: %.4f (n=%d)" % (rel, acc, mask.sum()))

    # Feature importance
    print("\n  Feature importance (top coefficients):")
    feature_names = ["cosine", "diff_norm", "head_norm", "child_norm", "dot_product"]
    feature_names += ["ch%d_cos" % i for i in range(config["n_channels"])]
    importances = np.abs(clf.coef_).mean(axis=0)
    top_idx = np.argsort(importances)[::-1][:10]
    for idx in top_idx:
        if idx < len(feature_names):
            print("    %s: %.4f" % (feature_names[idx], importances[idx]))

    # ============================================================
    # Test 3: Per-channel analysis
    # ============================================================
    print("\n" + "=" * 60)
    print("Test 3: Per-channel cosine analysis")
    print("=" * 60)

    # Which channels show the most difference between dep and non-dep?
    C = config["n_channels"]
    ch_offset = 5  # first 5 features are global
    dep_mask = np.array([l != "none" for l in y_all])

    print("  Channels with largest dep/non-dep cosine difference:")
    ch_diffs = []
    for ch in range(C):
        dep_ch_cos = X_all[dep_mask, ch_offset + ch]
        nodep_ch_cos = X_all[~dep_mask, ch_offset + ch]
        diff = dep_ch_cos.mean() - nodep_ch_cos.mean()
        ch_diffs.append((ch, diff, dep_ch_cos.mean(), nodep_ch_cos.mean()))

    ch_diffs.sort(key=lambda x: abs(x[1]), reverse=True)
    for ch, diff, dep_mean, nodep_mean in ch_diffs[:10]:
        print("    ch%02d: dep=%.4f, nodep=%.4f, diff=%.4f" % (ch, dep_mean, nodep_mean, diff))

    # ============================================================
    # Conclusion
    # ============================================================
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)

    significant_cosine = abs(d_all) > 0.3
    good_probe = probe_accuracy_4class > 0.5
    good_binary = probe_accuracy_binary > 0.65

    if significant_cosine and good_probe:
        conclusion = ("POSITIVE: Grade-2 direction (cosine) encodes dependency relations. "
                     "d=%.3f, 4-class=%.3f, binary=%.3f. "
                     "Dependency info is in grade-2 vector direction, not geometric product." % (d_all, probe_accuracy_4class, probe_accuracy_binary))
    elif good_probe or good_binary:
        conclusion = ("PARTIAL: Grade-2 features contain some dependency signal via direction/norm. "
                     "d=%.3f, 4-class=%.3f, binary=%.3f. "
                     "Per-channel cosine features help classification." % (d_all, probe_accuracy_4class, probe_accuracy_binary))
    else:
        conclusion = ("NEGATIVE: Grade-2 direction does not clearly encode dependencies either. "
                     "d=%.3f, 4-class=%.3f, binary=%.3f. "
                     "The Probing signal (0.84) may come from a different mechanism." % (d_all, probe_accuracy_4class, probe_accuracy_binary))

    print(conclusion)

    # Save results
    results.update({
        "experiment": "M v3: Grade-2 Cosine Similarity Analysis",
        "checkpoint": ckpt_path,
        "checkpoint_step": ckpt["step"],
        "n_dep_pairs": sum(len(v) for k, v in cosines_by_rel.items() if k != "none"),
        "n_nodep_pairs": len(cosines_by_rel["none"]),
        "ttest_per_rel": ttest_results,
        "ttest_diff_per_rel": ttest_diff_results,
        "probe_accuracy_4class": float(probe_accuracy_4class),
        "probe_accuracy_4class_std": float(scores.std()),
        "probe_accuracy_binary": float(probe_accuracy_binary),
        "dep_type_accuracy": dep_type_acc,
        "top_channels": [{"ch": ch, "diff": float(diff)} for ch, diff, _, _ in ch_diffs[:10]],
        "conclusion": conclusion,
    })

    os.makedirs("/data/biic/results", exist_ok=True)
    save_path = "/data/biic/results/exp_m_v3_cosine.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\nResults saved: %s" % save_path)


if __name__ == "__main__":
    run_experiment()
