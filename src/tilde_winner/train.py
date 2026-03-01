import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from tilde_winner.config import DataConfig, ModelConfig
from tilde_winner.data import sample_batch
from tilde_winner.model import DecoderLM
from tilde_winner.optim import MuonLike


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def final_token_accuracy(logits: torch.Tensor, targets: torch.Tensor, answer_pos: int) -> float:
    pred = logits[:, answer_pos, :].argmax(dim=-1)
    truth = targets[:, answer_pos]
    return float((pred == truth).float().mean().item())


@torch.no_grad()
def evaluate(
    model: DecoderLM,
    data_cfg: DataConfig,
    batch_size: int,
    eval_batches: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    ce_loss = 0.0
    aux_loss = 0.0
    acc = 0.0
    gate_mean = 0.0
    confidence_mean = 0.0

    for _ in range(eval_batches):
        x, y, answer_pos = sample_batch(data_cfg, batch_size=batch_size, device=device)
        logits, loss, aux, metrics = model(x, y)
        assert loss is not None
        ce_loss += float(loss.item())
        aux_loss += float(aux.item())
        acc += final_token_accuracy(logits, y, answer_pos)
        gate_mean += metrics["gate_mean"]
        confidence_mean += metrics["confidence_mean"]

    model.train()
    denom = float(eval_batches)
    return {
        "eval_ce": ce_loss / denom,
        "eval_aux": aux_loss / denom,
        "eval_retrieval_acc": acc / denom,
        "eval_gate_mean": gate_mean / denom,
        "eval_confidence_mean": confidence_mean / denom,
    }


def build_optimizer(model: nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    if args.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return MuonLike(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        ns_steps=args.muon_ns_steps,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline vs EGM Mistral-style model.")
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--variant", type=str, default="egm", choices=["baseline", "egm"])
    parser.add_argument("--optimizer", type=str, default="muon", choices=["adamw", "muon"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batches", type=int, default=50)
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--log-interval", type=int, default=25)

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--aux-weight", type=float, default=0.1)
    parser.add_argument("--muon-ns-steps", type=int, default=2)

    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-kv-heads", type=int, default=4)
    parser.add_argument("--ffn-mult", type=float, default=3.5)
    parser.add_argument("--dropout", type=float, default=0.0)

    parser.add_argument("--n-keys", type=int, default=64)
    parser.add_argument("--num-pairs", type=int, default=32)
    return parser.parse_args()


def run(args: argparse.Namespace) -> Path:
    set_seed(args.seed)
    device = torch.device(args.device)

    data_cfg = DataConfig(n_keys=args.n_keys, num_pairs=args.num_pairs)
    model_cfg = ModelConfig(
        vocab_size=data_cfg.vocab_size,
        max_seq_len=data_cfg.seq_len - 1,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        ffn_mult=args.ffn_mult,
        dropout=args.dropout,
        variant=args.variant,
    )
    model = DecoderLM(model_cfg).to(device)
    optimizer = build_optimizer(model, args)

    timestamp = int(time.time())
    run_name = args.run_name or f"{args.variant}_{args.optimizer}_{timestamp}"
    out_dir = Path("runs") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    metrics_file = out_dir / "metrics.csv"
    with metrics_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "step",
                "train_ce",
                "train_aux",
                "train_total",
                "train_retrieval_acc",
                "eval_ce",
                "eval_aux",
                "eval_retrieval_acc",
                "eval_gate_mean",
                "eval_confidence_mean",
            ]
        )

    best_eval_acc = -1.0
    for step in range(1, args.steps + 1):
        x, y, answer_pos = sample_batch(data_cfg, batch_size=args.batch_size, device=device)
        logits, ce, aux, _ = model(x, y)
        assert ce is not None

        total = ce + (args.aux_weight * aux if args.variant == "egm" else 0.0)
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
        optimizer.step()

        if step % args.log_interval == 0:
            train_acc = final_token_accuracy(logits, y, answer_pos)
            print(
                f"step={step} ce={ce.item():.4f} aux={aux.item():.4f} "
                f"total={float(total.item()):.4f} retrieval_acc={train_acc:.4f}"
            )

        if step % args.eval_interval == 0 or step == args.steps:
            eval_metrics = evaluate(
                model=model,
                data_cfg=data_cfg,
                batch_size=args.batch_size,
                eval_batches=args.eval_batches,
                device=device,
            )
            train_acc = final_token_accuracy(logits, y, answer_pos)
            row = [
                step,
                float(ce.item()),
                float(aux.item()),
                float(total.item()),
                train_acc,
                eval_metrics["eval_ce"],
                eval_metrics["eval_aux"],
                eval_metrics["eval_retrieval_acc"],
                eval_metrics["eval_gate_mean"],
                eval_metrics["eval_confidence_mean"],
            ]
            with metrics_file.open("a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(row)

            print(
                f"[eval] step={step} eval_ce={eval_metrics['eval_ce']:.4f} "
                f"eval_retrieval_acc={eval_metrics['eval_retrieval_acc']:.4f} "
                f"gate={eval_metrics['eval_gate_mean']:.4f}"
            )

            if eval_metrics["eval_retrieval_acc"] > best_eval_acc:
                best_eval_acc = eval_metrics["eval_retrieval_acc"]
                torch.save(
                    {
                        "model_cfg": model_cfg.__dict__,
                        "model_state_dict": model.state_dict(),
                        "step": step,
                        "best_eval_retrieval_acc": best_eval_acc,
                    },
                    out_dir / "best.pt",
                )

    print(f"run complete: {out_dir}")
    return out_dir


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
