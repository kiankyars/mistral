# Entropy-Gated Residual Mixing for Mistral-Style Decoders

This project targets **Tilde Research - Best Architectural Modification** with a concrete structural change:

- Baseline Mistral-style block: pre-norm -> attention residual -> MLP residual
- Proposed block (**EGM**): pre-norm -> attention + MLP in parallel -> token-wise gated residual mix

The gate is conditioned on **attention confidence** (1 - normalized attention entropy), so the model learns:

- high-confidence retrieval tokens -> favor attention update
- low-confidence tokens -> favor MLP/world-modeling update

## Why this is a strong challenge submission

- **Principled mechanism**: residual routing is tied to information-theoretic confidence
- **Real architecture change**: not prompt engineering, not only hyperparameters
- **Training dynamic change**: auxiliary gate-confidence alignment loss + optional Muon-like optimizer
- **Ablation-ready**: baseline vs EGM runs out of the box

## Benchmark

To demonstrate the behavioral change fast, we use a synthetic long-context retrieval task:

- sequence contains many key-value pairs
- final query asks for value of one earlier key
- metric = final-token retrieval accuracy

This isolates memory/retrieval behavior without data wrangling noise.

## Quickstart (uv)

```bash
uv sync
```

### Train baseline

```bash
uv run tilde-train --variant baseline --optimizer adamw --steps 3000
```

### Train modified architecture (EGM)

```bash
uv run tilde-train --variant egm --optimizer muon --steps 3000
```

### Run ablation end-to-end

```bash
uv run tilde-ablate --steps 3000 --optimizer muon
```

Outputs are saved under `runs/<run_name>/`:

- `metrics.csv`
- `best.pt`
- `config.json`

## Suggested hackathon demo script

1. Explain baseline block vs EGM block in one slide.
2. Run `tilde-ablate` with fixed seed.
3. Show retrieval accuracy gap and gate statistics from `metrics.csv`.
4. Explain observed behavior:
   - gate mean tracks confidence,
   - model routes retrieval-heavy positions toward attention.

## Repo structure

- `src/tilde_winner/model.py`: baseline + EGM transformer blocks
- `src/tilde_winner/optim.py`: Muon-like gradient orthogonalization optimizer
- `src/tilde_winner/data.py`: synthetic long-context retrieval generator
- `src/tilde_winner/train.py`: training/eval loop
- `src/tilde_winner/ablate.py`: baseline vs EGM runner
- `submission_pitch.md`: short submission narrative for judges
