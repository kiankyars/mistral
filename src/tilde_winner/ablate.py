import argparse
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline vs EGM ablation.")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batches", type=int, default=50)
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--optimizer", type=str, default="muon", choices=["adamw", "muon"])
    parser.add_argument("--n-keys", type=int, default=64)
    parser.add_argument("--num-pairs", type=int, default=32)
    return parser.parse_args()


def _run_variant(args: argparse.Namespace, variant: str) -> None:
    cmd = [
        sys.executable,
        "-m",
        "tilde_winner.train",
        "--variant",
        variant,
        "--optimizer",
        args.optimizer,
        "--steps",
        str(args.steps),
        "--batch-size",
        str(args.batch_size),
        "--eval-batches",
        str(args.eval_batches),
        "--eval-interval",
        str(args.eval_interval),
        "--seed",
        str(args.seed),
        "--run-name",
        f"{variant}_{args.optimizer}_seed{args.seed}",
        "--n-keys",
        str(args.n_keys),
        "--num-pairs",
        str(args.num_pairs),
    ]
    if args.device:
        cmd.extend(["--device", args.device])

    print("running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    _run_variant(args, "baseline")
    _run_variant(args, "egm")
    print("ablation complete. compare metrics under runs/")


if __name__ == "__main__":
    main()
