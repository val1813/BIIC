"""
Experiment M v2: Dependency Relation Geometric Product Analysis
Using TRAINED BIIC LM checkpoint (not random init)
Data: Real sentences from UD English EWT with gold dependency annotations

Three-level judgment:
1. t-test: dep vs non-dep grade-2 geometric product norms
2. Linear probe: classify dependency types from geometric product features
3. t-SNE visualization (saved as data for plotting)

Saves: /data/biic/results/exp_m_v2_dep_geometric.json
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

from clifford_cl41 import GRADE_SLICES, N_BLADES, geometric_product_fast
from token_to_ic import TokenToImmutableCore
from mutable_state import BIICLayer
from eraser_ops import GradeAwareEraser

device = torch.device("cpu")  # CPU experiment


# ============================================================
# Model Definition (must match checkpoint)
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
        self.kernel_size = kernel_size

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
        """Run encoder + blocks, return multivector [B, L, C, 32]"""
        mv = self.encoder(token_ids)
        mv = self.pos_enc(mv)
        mv_prior = mv.clone().detach()
        for block in self.blocks:
            mv = block(mv, mv_prior)
        return mv


# ============================================================
# UD Data Parsing
# ============================================================

def parse_conllu(path, max_sentences=2000):
    """Parse CoNLL-U file, extract sentences with dependency info"""
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
    """Load GPT-2 tokenizer for encoding"""
    from transformers import GPT2Tokenizer
    tok_path = "/data/biic/gpt2_tokenizer"
    if os.path.exists(tok_path):
        return GPT2Tokenizer.from_pretrained(tok_path)
    cache_path = "/data/biic/models_cache/AI-ModelScope/gpt2"
    if os.path.exists(cache_path):
        return GPT2Tokenizer.from_pretrained(cache_path)
    raise RuntimeError("No GPT-2 tokenizer found")


def extract_dep_pairs(sentences, target_rels=("nsubj", "obj", "amod")):
    """
    Extract token pairs with dependency relations.
    Returns: dep_pairs [(sent_idx, tok_i, tok_j, rel)], nodep_pairs, cross_pairs
    """
    dep_pairs = []
    nodep_pairs = []
    cross_pairs = []

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

    # Cross-sentence pairs (control group)
    rng = random.Random(12345)
    for _ in range(min(len(dep_pairs), 500)):
        s1 = rng.randint(0, len(sentences) - 1)
        s2 = rng.randint(0, len(sentences) - 1)
        while s2 == s1:
            s2 = rng.randint(0, len(sentences) - 1)
        i = rng.randint(0, len(sentences[s1]["tokens"]) - 1)
        j = rng.randint(0, len(sentences[s2]["tokens"]) - 1)
        cross_pairs.append((s1, s2, i, j, "cross"))

    return dep_pairs, nodep_pairs, cross_pairs


# ============================================================
# Geometric Product Computation
# ============================================================

def compute_grade2_geometric_product(mv_i, mv_j):
    """
    Compute geometric product of grade-2 components.
    mv_i, mv_j: [C, 32] multivectors for two tokens
    Returns: grade-0 scalar part [C], grade-2 vector part [C, 10]
    """
    g2_start, g2_end = GRADE_SLICES[2]

    g2_i = mv_i[:, g2_start:g2_end]
    g2_j = mv_j[:, g2_start:g2_end]

    # Build full multivectors with only grade-2 nonzero
    mv_full_i = torch.zeros(mv_i.shape[0], N_BLADES, device=mv_i.device)
    mv_full_j = torch.zeros(mv_j.shape[0], N_BLADES, device=mv_j.device)
    mv_full_i[:, g2_start:g2_end] = g2_i
    mv_full_j[:, g2_start:g2_end] = g2_j

    # Geometric product
    gp = geometric_product_fast(mv_full_i, mv_full_j)  # [C, 32]

    # Extract grade-0 (scalar) and grade-2 parts
    g0_start, g0_end = GRADE_SLICES[0]
    gp_grade0 = gp[:, g0_start:g0_end].squeeze(-1)  # [C]
    gp_grade2 = gp[:, g2_start:g2_end]  # [C, 10]

    return gp_grade0, gp_grade2


# ============================================================
# Main Experiment
# ============================================================

def run_experiment():
    print("=" * 60)
    print("Experiment M v2: Dependency Geometric Product (Trained Model)")
    print("=" * 60)

    # Load checkpoint
    ckpt_path = "/data/biic/biic_lm/checkpoints/final.pt"
    print("Loading checkpoint:", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    print("Config: n_channels=%d, n_layers=%d, step=%d" % (config["n_channels"], config["n_layers"], ckpt["step"]))

    # Build model and load weights
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
    dep_pairs, nodep_pairs, cross_pairs = extract_dep_pairs(sentences)
    print("  nsubj/obj/amod pairs: %d" % len(dep_pairs))
    print("  non-dep pairs: %d" % len(nodep_pairs))
    print("  cross-sentence pairs: %d" % len(cross_pairs))

    # Encode sentences and compute geometric products
    print("\nEncoding sentences and computing geometric products...")

    encoded_cache = {}

    def get_encoded_sentence(s_idx):
        if s_idx not in encoded_cache:
            tokens = sentences[s_idx]["tokens"]
            text = " ".join(tokens)
            input_ids = tokenizer.encode(text, add_special_tokens=False)
            # Map word positions to token positions (first subword)
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

    dep_gp_norms = []
    dep_gp_features = []
    dep_labels = []
    nodep_gp_norms = []
    nodep_gp_features = []
    cross_gp_norms = []

    # Process dependency pairs
    print("  Processing dep pairs...")
    for s_idx, head_pos, child_pos, rel in dep_pairs:
        try:
            enc = get_encoded_sentence(s_idx)
            wtt = enc["word_to_token"]
            if head_pos >= len(wtt) or child_pos >= len(wtt):
                continue
            tok_i = wtt[head_pos]
            tok_j = wtt[child_pos]
            if tok_i >= enc["n_tokens"] or tok_j >= enc["n_tokens"]:
                continue
            mv_i = enc["mv"][tok_i]  # [C, 32]
            mv_j = enc["mv"][tok_j]  # [C, 32]
            gp_g0, gp_g2 = compute_grade2_geometric_product(mv_i, mv_j)
            norm = gp_g2.norm().item()
            dep_gp_norms.append(norm)
            feat = torch.cat([gp_g0, gp_g2.norm(dim=-1)]).numpy()
            dep_gp_features.append(feat)
            dep_labels.append(rel)
        except Exception:
            continue

    # Process non-dep pairs
    print("  Processing non-dep pairs...")
    for s_idx, i, j, _ in nodep_pairs:
        try:
            enc = get_encoded_sentence(s_idx)
            wtt = enc["word_to_token"]
            if i >= len(wtt) or j >= len(wtt):
                continue
            tok_i = wtt[i]
            tok_j = wtt[j]
            if tok_i >= enc["n_tokens"] or tok_j >= enc["n_tokens"]:
                continue
            mv_i = enc["mv"][tok_i]
            mv_j = enc["mv"][tok_j]
            gp_g0, gp_g2 = compute_grade2_geometric_product(mv_i, mv_j)
            norm = gp_g2.norm().item()
            nodep_gp_norms.append(norm)
            feat = torch.cat([gp_g0, gp_g2.norm(dim=-1)]).numpy()
            nodep_gp_features.append(feat)
        except Exception:
            continue

    # Process cross-sentence pairs
    print("  Processing cross-sentence pairs...")
    for s1, s2, i, j, _ in cross_pairs[:500]:
        try:
            enc1 = get_encoded_sentence(s1)
            enc2 = get_encoded_sentence(s2)
            wtt1 = enc1["word_to_token"]
            wtt2 = enc2["word_to_token"]
            if i >= len(wtt1) or j >= len(wtt2):
                continue
            tok_i = wtt1[i]
            tok_j = wtt2[j]
            if tok_i >= enc1["n_tokens"] or tok_j >= enc2["n_tokens"]:
                continue
            mv_i = enc1["mv"][tok_i]
            mv_j = enc2["mv"][tok_j]
            gp_g0, gp_g2 = compute_grade2_geometric_product(mv_i, mv_j)
            norm = gp_g2.norm().item()
            cross_gp_norms.append(norm)
        except Exception:
            continue

    # ============================================================
    # Judgment 1: t-test on norms
    # ============================================================
    print("\n" + "=" * 60)
    print("Judgment 1: t-test on grade-2 geometric product norms")
    print("=" * 60)

    dep_arr = np.array(dep_gp_norms)
    nodep_arr = np.array(nodep_gp_norms)
    cross_arr = np.array(cross_gp_norms)

    t_stat, p_value = stats.ttest_ind(dep_arr, nodep_arr)
    pooled_std = np.sqrt((dep_arr.std()**2 + nodep_arr.std()**2) / 2)
    cohens_d = (dep_arr.mean() - nodep_arr.mean()) / pooled_std if pooled_std > 0 else 0

    print("  Dep pairs: n=%d, mean=%.4f, std=%.4f" % (len(dep_arr), dep_arr.mean(), dep_arr.std()))
    print("  Non-dep pairs: n=%d, mean=%.4f, std=%.4f" % (len(nodep_arr), nodep_arr.mean(), nodep_arr.std()))
    print("  Cross pairs: n=%d, mean=%.4f, std=%.4f" % (len(cross_arr), cross_arr.mean(), cross_arr.std()))
    print("  t-statistic: %.4f" % t_stat)
    print("  p-value: %.6f" % p_value)
    print("  Cohen's d: %.4f" % cohens_d)
    print("  Significant (p<0.05): %s" % (p_value < 0.05))

    # ============================================================
    # Judgment 2: Linear probe for dependency type classification
    # ============================================================
    print("\n" + "=" * 60)
    print("Judgment 2: Linear probe classification")
    print("=" * 60)

    X_dep = np.array(dep_gp_features)
    y_dep_types = np.array(dep_labels)

    # 4-class: nsubj, obj, amod, none
    n_nodep_use = min(len(nodep_gp_features), len(dep_gp_features))
    X_nodep = np.array(nodep_gp_features[:n_nodep_use])
    X_all = np.vstack([X_dep, X_nodep])
    y_all = np.concatenate([y_dep_types, np.array(["none"] * len(X_nodep))])

    label_map = {"nsubj": 0, "obj": 1, "amod": 2, "none": 3}
    y_encoded = np.array([label_map.get(l, 3) for l in y_all])

    clf = LogisticRegression(max_iter=1000, random_state=42, multi_class="multinomial")
    scores = cross_val_score(clf, X_all, y_encoded, cv=5, scoring="accuracy")
    probe_accuracy = scores.mean()

    print("  Features shape: %s" % str(X_all.shape))
    print("  Classes: %s" % str(label_map))
    print("  5-fold CV accuracy: %.4f (+/- %.4f)" % (probe_accuracy, scores.std()))
    print("  Target: > 0.75")

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

    # ============================================================
    # Judgment 3: t-SNE data
    # ============================================================
    print("\n" + "=" * 60)
    print("Judgment 3: t-SNE embedding")
    print("=" * 60)

    n_tsne = min(500, len(X_all))
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X_all[:n_tsne])
    y_tsne = y_all[:n_tsne]

    print("  t-SNE computed on %d samples" % n_tsne)
    print("  Checking cluster separation...")

    from scipy.spatial.distance import cdist
    sep_scores = {}
    for rel in ["nsubj", "obj", "amod", "none"]:
        mask = y_tsne == rel
        if mask.sum() > 5:
            intra = cdist(X_tsne[mask], X_tsne[mask]).mean()
            inter = cdist(X_tsne[mask], X_tsne[~mask]).mean()
            sep_scores[rel] = float(inter / intra) if intra > 0 else 0
            print("    %s: separation ratio = %.3f" % (rel, sep_scores[rel]))

    # ============================================================
    # Conclusion
    # ============================================================
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)

    significant = p_value < 0.05
    strong_effect = abs(cohens_d) > 0.5
    good_probe = probe_accuracy > 0.75

    if significant and strong_effect and good_probe:
        conclusion = ("POSITIVE: Grade-2 geometric product encodes dependency relations. "
                     "p=%.6f, d=%.3f, probe_acc=%.3f. "
                     "This confirms BIIC algebraic structure captures linguistic dependencies." % (p_value, cohens_d, probe_accuracy))
    elif significant and (strong_effect or good_probe):
        conclusion = ("PARTIAL: Some evidence for dependency encoding. "
                     "p=%.6f, d=%.3f, probe_acc=%.3f. "
                     "Effect exists but may not be strong enough for definitive claim." % (p_value, cohens_d, probe_accuracy))
    else:
        conclusion = ("NEGATIVE: Grade-2 geometric product does not clearly encode dependencies. "
                     "p=%.6f, d=%.3f, probe_acc=%.3f. "
                     "Training did not induce dependency-specific geometric structure." % (p_value, cohens_d, probe_accuracy))

    print(conclusion)

    # Save results
    results = {
        "experiment": "M v2: Dependency Geometric Product (Trained Model)",
        "checkpoint": ckpt_path,
        "checkpoint_step": ckpt["step"],
        "n_dep_pairs": len(dep_gp_norms),
        "n_nodep_pairs": len(nodep_gp_norms),
        "n_cross_pairs": len(cross_gp_norms),
        "ttest_pvalue": float(p_value),
        "ttest_statistic": float(t_stat),
        "cohens_d": float(cohens_d),
        "dep_norm_mean": float(dep_arr.mean()),
        "dep_norm_std": float(dep_arr.std()),
        "nodep_norm_mean": float(nodep_arr.mean()),
        "nodep_norm_std": float(nodep_arr.std()),
        "cross_norm_mean": float(cross_arr.mean()) if len(cross_arr) > 0 else None,
        "probe_accuracy": float(probe_accuracy),
        "probe_accuracy_std": float(scores.std()),
        "dep_type_accuracy": dep_type_acc,
        "tsne_separation": sep_scores,
        "conclusion": conclusion,
    }

    os.makedirs("/data/biic/results", exist_ok=True)
    save_path = "/data/biic/results/exp_m_v2_dep_geometric.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\nResults saved: %s" % save_path)


if __name__ == "__main__":
    run_experiment()
