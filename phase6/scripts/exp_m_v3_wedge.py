"""
Experiment M v3-wedge: Grade-1 Wedge Product (Commutator) Analysis
From mathematical first principles:
  - grade-1 vectors represent token directions in Cl(4,1)
  - wedge product g1_i ∧ g1_j = the relationship PLANE between two tokens
  - commutator [g1_i, g1_j] = 2*(g1_i ∧ g1_j) for grade-1 vectors
  - different dependency types should correspond to different wedge directions

This is the mathematically correct operation for testing dependency encoding.

Uses trained BIIC LM checkpoint.
Saves: /data/biic/results/exp_m_v3_wedge.json
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
from sklearn.manifold import TSNE
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
# Model Definition (same as before)
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
# Wedge Product (the mathematically correct operation)
# ============================================================

def compute_wedge_product(g1_i, g1_j):
    """
    Compute wedge product of two grade-1 vectors.
    g1_i, g1_j: [C, 5] (grade-1 components: e1,e2,e3,e4,e5)
    Returns: [C, 10] (grade-2 bivector: the relationship plane)

    Wedge product = antisymmetric part of geometric product
    (ei ∧ ej) = ei*ej for i<j (antisymmetric)

    Blade ordering in Cl(4,1) grade-2:
    [0]=e12, [1]=e13, [2]=e14, [3]=e15
    [4]=e23, [5]=e24, [6]=e25
    [7]=e34, [8]=e35
    [9]=e45
    """
    C = g1_i.shape[0]
    wedge = torch.zeros(C, 10, device=g1_i.device)

    # pairs (i,j) with i<j, mapping to blade index
    pairs = [(0, 1), (0, 2), (0, 3), (0, 4),
             (1, 2), (1, 3), (1, 4),
             (2, 3), (2, 4),
             (3, 4)]

    for k, (i, j) in enumerate(pairs):
        # a_i * b_j - a_j * b_i (antisymmetric = wedge)
        wedge[:, k] = g1_i[:, i] * g1_j[:, j] - g1_i[:, j] * g1_j[:, i]

    return wedge  # [C, 10]


def compute_commutator_norm(g1_i, g1_j):
    """
    Commutator [a,b] = ab - ba = 2*(a∧b) for grade-1 vectors.
    Returns scalar: mean norm across channels.
    """
    wedge = compute_wedge_product(g1_i, g1_j)
    # commutator = 2 * wedge, norm = 2 * wedge_norm
    return (2 * wedge).norm(dim=-1).mean().item()


# ============================================================
# UD Data Parsing
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
    print("Experiment M v3-wedge: Grade-1 Wedge Product Analysis")
    print("Mathematical basis: dependency = wedge product of grade-1 vectors")
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
    print("\nEncoding sentences and computing wedge products...")
    encoded_cache = {}
    g1_start, g1_end = GRADE_SLICES[1]  # grade-1: indices 1:6

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
                mv = model.encode(ids_tensor)
            encoded_cache[s_idx] = {"mv": mv[0], "word_to_token": word_to_token, "n_tokens": mv.shape[1]}
        return encoded_cache[s_idx]

    def get_grade1(enc, word_pos):
        """Get grade-1 vector [C, 5] for a word position"""
        wtt = enc["word_to_token"]
        if word_pos >= len(wtt):
            return None
        tok_pos = wtt[word_pos]
        if tok_pos >= enc["n_tokens"]:
            return None
        mv_tok = enc["mv"][tok_pos]  # [C, 32]
        return mv_tok[:, g1_start:g1_end]  # [C, 5]

    # ============================================================
    # Compute wedge products for all pairs
    # ============================================================

    wedge_norms_by_rel = defaultdict(list)
    commutator_norms_by_rel = defaultdict(list)
    wedge_features = []
    wedge_labels = []

    print("  Processing dep pairs...")
    for s_idx, head_pos, child_pos, rel in dep_pairs:
        try:
            enc = get_encoded_sentence(s_idx)
            g1_head = get_grade1(enc, head_pos)
            g1_child = get_grade1(enc, child_pos)
            if g1_head is None or g1_child is None:
                continue

            # Wedge product: the relationship plane
            wedge = compute_wedge_product(g1_head, g1_child)  # [C, 10]
            wedge_norm = wedge.norm(dim=-1).mean().item()
            comm_norm = compute_commutator_norm(g1_head, g1_child)

            wedge_norms_by_rel[rel].append(wedge_norm)
            commutator_norms_by_rel[rel].append(comm_norm)

            # Feature: flattened wedge product (or summary stats)
            # Use per-channel norms + overall direction
            wedge_flat = wedge.reshape(-1).numpy()  # [C*10 = 320]
            wedge_features.append(wedge_flat)
            wedge_labels.append(rel)
        except Exception:
            continue

    print("  Processing non-dep pairs...")
    for s_idx, i, j, _ in nodep_pairs:
        try:
            enc = get_encoded_sentence(s_idx)
            g1_i = get_grade1(enc, i)
            g1_j = get_grade1(enc, j)
            if g1_i is None or g1_j is None:
                continue

            wedge = compute_wedge_product(g1_i, g1_j)
            wedge_norm = wedge.norm(dim=-1).mean().item()
            comm_norm = compute_commutator_norm(g1_i, g1_j)

            wedge_norms_by_rel["none"].append(wedge_norm)
            commutator_norms_by_rel["none"].append(comm_norm)

            wedge_flat = wedge.reshape(-1).numpy()
            wedge_features.append(wedge_flat)
            wedge_labels.append("none")
        except Exception:
            continue

    # ============================================================
    # Analysis 1: Commutator/Wedge norm comparison
    # ============================================================
    print("\n" + "=" * 60)
    print("Analysis 1: Wedge product norm by relation type")
    print("(Commutator norm = 2 * wedge norm = relationship strength)")
    print("=" * 60)

    results = {"wedge_norm_by_rel": {}, "commutator_norm_by_rel": {}}

    for rel in ["nsubj", "obj", "amod", "none"]:
        w_arr = np.array(wedge_norms_by_rel[rel])
        c_arr = np.array(commutator_norms_by_rel[rel])
        if len(w_arr) > 0:
            results["wedge_norm_by_rel"][rel] = {
                "mean": float(w_arr.mean()), "std": float(w_arr.std()), "n": len(w_arr)
            }
            results["commutator_norm_by_rel"][rel] = {
                "mean": float(c_arr.mean()), "std": float(c_arr.std()), "n": len(c_arr)
            }
            print("  %s: wedge_norm=%.4f+/-%.4f, comm_norm=%.4f+/-%.4f (n=%d)" % (
                rel, w_arr.mean(), w_arr.std(), c_arr.mean(), c_arr.std(), len(w_arr)))

    # t-tests: dep vs non-dep
    print("\n  t-tests (wedge norm): dep type vs non-dep")
    none_wedge = np.array(wedge_norms_by_rel["none"])
    ttest_results = {}
    for rel in ["nsubj", "obj", "amod"]:
        rel_wedge = np.array(wedge_norms_by_rel[rel])
        if len(rel_wedge) > 10:
            t_stat, p_val = stats.ttest_ind(rel_wedge, none_wedge)
            pooled_std = np.sqrt((rel_wedge.std()**2 + none_wedge.std()**2) / 2)
            d = (rel_wedge.mean() - none_wedge.mean()) / pooled_std if pooled_std > 0 else 0
            ttest_results[rel] = {"t_stat": float(t_stat), "p_value": float(p_val), "cohens_d": float(d)}
            print("    %s vs none: t=%.3f, p=%.6f, d=%.4f %s" % (
                rel, t_stat, p_val, d, "***" if p_val < 0.001 else "*" if p_val < 0.05 else ""))

    # All dep vs none
    all_dep_wedge = np.concatenate([np.array(wedge_norms_by_rel[r]) for r in ["nsubj", "obj", "amod"]])
    t_all, p_all = stats.ttest_ind(all_dep_wedge, none_wedge)
    pooled_all = np.sqrt((all_dep_wedge.std()**2 + none_wedge.std()**2) / 2)
    d_all = (all_dep_wedge.mean() - none_wedge.mean()) / pooled_all if pooled_all > 0 else 0
    print("    ALL_DEP vs none: t=%.3f, p=%.6f, d=%.4f" % (t_all, p_all, d_all))
    results["ttest_all_dep_vs_none_wedge"] = {"t_stat": float(t_all), "p_value": float(p_all), "cohens_d": float(d_all)}

    # ============================================================
    # Analysis 2: Linear probe on wedge product features
    # ============================================================
    print("\n" + "=" * 60)
    print("Analysis 2: Linear probe on wedge product vectors")
    print("=" * 60)

    X_all = np.array(wedge_features)
    y_all = np.array(wedge_labels)

    # 4-class
    label_map = {"nsubj": 0, "obj": 1, "amod": 2, "none": 3}
    y_encoded = np.array([label_map.get(l, 3) for l in y_all])

    clf = LogisticRegression(max_iter=2000, random_state=42)
    scores = cross_val_score(clf, X_all, y_encoded, cv=5, scoring="accuracy")
    probe_4class = scores.mean()
    print("  4-class accuracy: %.4f (+/- %.4f)" % (probe_4class, scores.std()))
    print("  (random baseline: 0.25)")

    # 3-class (dep types only, excluding none)
    dep_mask = y_all != "none"
    if dep_mask.sum() > 100:
        X_dep = X_all[dep_mask]
        y_dep = y_all[dep_mask]
        label_map_3 = {"nsubj": 0, "obj": 1, "amod": 2}
        y_dep_enc = np.array([label_map_3[l] for l in y_dep])
        scores_3 = cross_val_score(clf, X_dep, y_dep_enc, cv=5, scoring="accuracy")
        probe_3class = scores_3.mean()
        print("  3-class (dep types only): %.4f (+/- %.4f)" % (probe_3class, scores_3.std()))
        print("  (random baseline: 0.33)")
    else:
        probe_3class = 0.0

    # Binary: dep vs non-dep
    y_binary = np.array([0 if l == "none" else 1 for l in y_all])
    scores_bin = cross_val_score(clf, X_all, y_binary, cv=5, scoring="accuracy")
    probe_binary = scores_bin.mean()
    print("  Binary (dep vs non-dep): %.4f (+/- %.4f)" % (probe_binary, scores_bin.std()))

    # Per-class
    clf.fit(X_all, y_encoded)
    dep_type_acc = {}
    for rel, idx in label_map.items():
        mask = y_encoded == idx
        if mask.sum() > 0:
            pred = clf.predict(X_all[mask])
            acc = (pred == idx).mean()
            dep_type_acc[rel] = float(acc)
            print("    %s: %.4f (n=%d)" % (rel, acc, mask.sum()))

    # ============================================================
    # Analysis 3: t-SNE of wedge product vectors
    # ============================================================
    print("\n" + "=" * 60)
    print("Analysis 3: t-SNE of wedge product space")
    print("=" * 60)

    n_tsne = min(800, len(X_all))
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X_all[:n_tsne])
    y_tsne = y_all[:n_tsne]

    # Separation metric
    from scipy.spatial.distance import cdist
    sep_scores = {}
    for rel in ["nsubj", "obj", "amod", "none"]:
        mask = y_tsne == rel
        if mask.sum() > 5:
            intra = cdist(X_tsne[mask], X_tsne[mask]).mean()
            inter = cdist(X_tsne[mask], X_tsne[~mask]).mean()
            sep_scores[rel] = float(inter / intra) if intra > 0 else 0
            print("  %s: separation ratio = %.3f (n=%d)" % (rel, sep_scores[rel], mask.sum()))

    # ============================================================
    # Conclusion
    # ============================================================
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)

    significant_norm = abs(d_all) > 0.3
    good_probe_4 = probe_4class > 0.5
    good_probe_3 = probe_3class > 0.5

    if significant_norm and good_probe_3:
        conclusion = ("POSITIVE: Grade-1 wedge product encodes dependency relations! "
                     "d=%.3f, 3-class=%.3f, 4-class=%.3f. "
                     "Dependency = wedge product direction in Cl(4,1). "
                     "This is the correct algebraic operation for relationship encoding." % (d_all, probe_3class, probe_4class))
    elif good_probe_4 or good_probe_3:
        conclusion = ("PARTIAL: Wedge product contains dependency signal. "
                     "d=%.3f, 3-class=%.3f, 4-class=%.3f. "
                     "Grade-1 wedge product partially encodes dependency type." % (d_all, probe_3class, probe_4class))
    else:
        conclusion = ("NEGATIVE: Grade-1 wedge product does not encode dependencies. "
                     "d=%.3f, 3-class=%.3f, 4-class=%.3f. "
                     "Neither geometric product nor wedge product captures dependency structure." % (d_all, probe_3class, probe_4class))

    print(conclusion)

    # Save
    results.update({
        "experiment": "M v3-wedge: Grade-1 Wedge Product Analysis",
        "checkpoint": ckpt_path,
        "checkpoint_step": ckpt["step"],
        "n_dep_pairs": sum(len(v) for k, v in wedge_norms_by_rel.items() if k != "none"),
        "n_nodep_pairs": len(wedge_norms_by_rel["none"]),
        "ttest_per_rel_wedge": ttest_results,
        "probe_accuracy_4class": float(probe_4class),
        "probe_accuracy_3class": float(probe_3class),
        "probe_accuracy_binary": float(probe_binary),
        "dep_type_accuracy": dep_type_acc,
        "tsne_separation": sep_scores,
        "conclusion": conclusion,
    })

    os.makedirs("/data/biic/results", exist_ok=True)
    save_path = "/data/biic/results/exp_m_v3_wedge.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\nResults saved: %s" % save_path)


if __name__ == "__main__":
    run_experiment()
