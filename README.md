# BIIC: Bio-Inspired Information Cell

A geometric algebra framework for lossless information representation in language models.

## The Problem

Current token representations in language models have a fundamental flaw: all semantic information lives in a single flat vector that gets progressively overwritten during inference. Residual connections mitigate but cannot solve this — they only add, never subtract. After dozens of transformer layers, the original meaning of a token is buried under accumulated intermediate states.

This causes three concrete failures:
1. **Information loss is irreversible** — deep layers cannot recover the original token semantics
2. **Information overload** — residual connections accumulate noise without any cleanup mechanism
3. **Long reasoning degrades** — performance drops systematically as inference depth increases

## Our Approach: Learn from DNA

DNA solves an analogous problem. A cell must simultaneously:
- Preserve its genome (identity) permanently across all operations
- Dynamically read/write epigenetic marks (current state) in response to signals
- Actively erase outdated marks via demethylation enzymes (TET/DNMT)

We map this three-layer architecture to a mathematical structure:

| DNA Layer | BIIC Component | Mathematical Realization |
|-----------|---------------|--------------------------|
| Genome (immutable) | Immutable Core | Grade-0 scalar (algebraically invariant) |
| Epigenome (read/write/erase) | Mutable State | Grade-1~4 equivariant components |
| Demethylase enzymes | GradeAwareEraser | Selective decay operator |

The key insight: **Clifford geometric algebra Cl(4,1) provides both invariant and equivariant quantities in the same algebraic structure**, with the invariance guaranteed by theorem, not engineering.

## Core Hypothesis

> A language information carrier based on Clifford algebra, where grade-0 is strictly invariant under all inference transformations and grade-1~4 carry independent equivariant semantics, outperforms flat token embeddings in information preservation and long-context reasoning.

## Minimum Experiment Plan

We validate this hypothesis in four phases:

| Phase | Goal | Status |
|-------|------|--------|
| Phase 1 | Mathematical verification of Cl(4,1) properties | ✅ Complete (3 seeds, 10/10 pass) |
| Phase 2 | Encoding-decoding pipeline validation | ✅ Complete (3 seeds, all pass) |
| Phase 3 | 6-group comparative experiment (H1/H2/H3 tests) | 🔄 Running |
| Phase 4 | MVP language model with SlowFast architecture | 📋 Planned (dry run passed) |

## Results So Far

### Phase 1: Grade-0 Invariance is Real

After 100 consecutive sandwich product transformations:

| Metric | Value (mean ± std, 3 seeds) |
|--------|----------------------------|
| Grade-0 invariance error | 6.56×10⁻⁶ ± 4.95×10⁻⁶ |
| Grade-5 invariance error | 5.12×10⁻⁶ ± 3.81×10⁻⁶ |
| Grade-1 equivariance error | 1.15×10⁻⁶ ± 8.2×10⁻⁷ |
| Multi-channel leakage | 0.0 (exact) |
| Eraser preserves grade-0 | 0.0 (exact) |

Grade-0 stays invariant to machine precision. This is a mathematical guarantee, not an engineering approximation.

### Phase 2: Equivariant Grades Carry Independent Semantics

| Metric | Value |
|--------|-------|
| All-grade decoding vs grade-0 only | 0.006 vs 0.032 loss (5.3× better) |
| Grade-0 token discrimination | cosine similarity 0.029 ± 0.013 |
| Grade norm separation (std) | 9.23 ± 0.08 |
| Grade-0 change after 6 inference layers | 0.0 (exact) |
| Decoder overfit improvement | 99.83% ± 0.001% |

The equivariant components (grade-1~4) carry semantic information that the invariant core alone cannot provide. Different tokens achieve near-orthogonal grade-0 representations (cos_sim ≈ 0.03).

### Phase 4 Dry Run: Architecture Validated

| Check | Result |
|-------|--------|
| Forward pass | ✅ |
| Grade-0 preserved during training | ✅ (change = 0.0) |
| Gradient flows to all components | ✅ |
| DualCodebook A/B both contribute | ✅ |
| Training converges (100 steps) | ✅ (34.6% improvement) |
| Checkpoint save/load | ✅ |

Full-scale config: 37M params, 4GB peak VRAM, ~0.9s/step on RTX 4090.

## What Comes Next

**Phase 3** (running now): Six-group comparison testing three hypotheses:
- H1: Does geometric structure itself matter, or just orthogonal constraints?
- H2: Do equivariant grades have independent value, or is it just higher dimensionality?
- H3: Does the Eraser actually control information entropy in long sequences?

**Phase 4** (after Phase 3): Train a minimal language model with:
- DualCodebookDecoder (separate codebooks for invariant/equivariant paths)
- SlowFast architecture (slow network every K steps, fast network every step)
- No residual connections (stability from algebraic invariance)

## If This Works

If Phase 3+4 confirm the hypotheses:

1. **Lossless long-context** — grade-0 preserves original semantics regardless of inference depth
2. **No KV cache needed** — mutable state replaces key-value storage
3. **Built-in interpretability** — grade decomposition reveals what the model "remembers" vs "is thinking about"
4. **O(L) complexity** — slow-fast separation eliminates quadratic attention cost
5. **Natural multimodal alignment** — different modalities share the same algebraic space, grade-0 is directly comparable

## Repository Structure

```
biic-repo/
├── README.md
├── src/
│   ├── clifford_cl41.py          # Cl(4,1) geometric algebra implementation
│   ├── rotor_utils.py            # Rotor generation, sandwich product, stabilization
│   ├── eraser_ops.py             # GradeAwareEraser (selective forgetting)
│   ├── token_to_ic.py            # Token → Immutable Core encoder
│   ├── all_grade_decoder.py      # AllGradeDecoder (gated multi-grade)
│   ├── mutable_state.py          # BIICLayer (Writer + Eraser)
│   └── biic_loss.py              # Staged auxiliary loss with annealing
├── tests/
│   ├── test_phase1.py            # 10 mathematical verification tests
│   ├── test_decoder_basic.py     # Decoder overfit + gradient + ablation
│   ├── test_encoder.py           # Grade separation + token discrimination
│   └── test_full_pipeline.py     # End-to-end pipeline + Eraser effectiveness
├── results/
│   ├── phase1/
│   │   └── phase1_results.json   # Full Phase 1 data (3 seeds × 10 tests)
│   └── phase2/
│       └── phase2_results.json   # Full Phase 2 data (3 seeds × 5 tests)
├── figures/
│   ├── fig1_grade0_invariance.png
│   ├── fig2_decoder_loss_curve.png
│   ├── fig3_grade_norm_distribution.png
│   └── fig4_token_discrimination.png
├── requirements.txt
└── LICENSE
```

## Quick Start

```bash
pip install torch numpy scipy matplotlib

# Run Phase 1 verification (CPU, ~2 minutes)
cd tests
python test_phase1.py

# Run Phase 2 pipeline tests (CPU, ~10 minutes)
python test_decoder_basic.py
python test_encoder.py
python test_full_pipeline.py
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- NumPy
- SciPy (for statistical tests)
- Matplotlib (for figure generation)

No GPU required for Phase 1+2 verification. Phase 3+4 use a single RTX 4090.

## References

- Hestenes & Sobczyk, 1984. *Clifford Algebra to Geometric Calculus*
- Brehmer et al., 2023. Geometric Algebra Transformer (GATr). NeurIPS 2023. [arXiv:2305.18415](https://arxiv.org/abs/2305.18415)
- Li et al., 2025. Versor: A Geometric Sequence Architecture. [arXiv:2602.10195](https://arxiv.org/abs/2602.10195)
- Author et al., 2025. Toward a Functional Geometric Algebra for Natural Language Semantics. [arXiv:2604.25902](https://arxiv.org/abs/2604.25902)
- Author et al., 2025. All You Need is Geometric Algebra (CliffordNet). [arXiv:2601.06793](https://arxiv.org/abs/2601.06793)
- Wu & Zhang, 2017. TET-mediated active DNA demethylation. *Nature Reviews Genetics*
- Zou et al., 2023. Representation Engineering. [arXiv:2310.01405](https://arxiv.org/abs/2310.01405)

## License

MIT

## Citation

```bibtex
@misc{huang2025biic,
  title={Bio-Inspired Information Cell: A Geometric Algebra Framework for Lossless Information Representation in Language Models},
  author={Huang, Zhongchang},
  year={2025},
  note={Experiments in progress. Phase 1-2 complete, Phase 3-4 ongoing.}
}
```
