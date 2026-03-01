# Tilde Challenge Submission Pitch

## Project
**Entropy-Gated Residual Mixing (EGM) for Mistral-style decoder transformers**

## Problem framing
Standard decoder blocks apply fixed residual paths:

- attention update is always added
- MLP update is always added

But token roles differ across a sequence. Some positions are explicit retrieval problems (copying from context), while others are synthesis/generalization steps. A fixed residual recipe is suboptimal.

## Architectural modification
I replace the standard sequential residual updates with a **single routed residual mix**:

1. Compute attention and MLP outputs in parallel from the same RMS-normalized stream.
2. Estimate attention confidence per token from normalized attention entropy.
3. Learn a token-wise gate:
   - input: hidden state + confidence
   - output: scalar in [0, 1]
4. Residual update:
   - `x <- x + s_l * (g * attn_out + (1 - g) * mlp_out)`

Where `s_l` is a learned per-layer residual scale.

## Training dynamics modification
I add an auxiliary objective aligning gate output with confidence:

- `L_aux = MSE(g, confidence.detach())`
- Total loss: `L = L_ce + lambda * L_aux`

I include an optional Muon-like optimizer (Newton-Schulz orthogonalized matrix gradients).

## Predicted behavioral change
- Better long-context retrieval: confident attention heads get amplified where copying is needed.
- Better compositional robustness: uncertain retrieval contexts shift capacity to MLP transforms.
- Improved optimization stability: orthogonalized updates reduce pathological matrix updates.

## Experimental setup
- Mistral-style decoder components: RMSNorm, RoPE, causal attention, SwiGLU, GQA-like kv grouping.
- Synthetic long-context key-value retrieval benchmark.
- Main metric: final-token retrieval accuracy.
- Ablation: baseline block vs EGM block under matched compute.
