#!/usr/bin/env python3
"""diagnose_training.py - why does one seed in nine never learn?

tools/train_nrr.py refuses a run whose loss barely moves, which is how this was noticed: 1 of 9 seeds
trained for 60 epochs and came out at the untrained-model loss, with the residual at 4e-05 of the bilinear
baseline. A gate that refuses it is necessary but not sufficient - a one-in-nine failure rate is a defect in
the training setup, not a property to live with.

The first explanation (a ReLU driven negative across a channel, zeroing the features that feed the
zero-initialised output convolution, and with them every gradient) was fixed with leaky activations and
verified on one seed. A later seed stalled anyway, so that explanation was at best incomplete, and this
measures instead of guessing: it trains a stalling seed and a healthy one side by side, printing per epoch
the loss, the output convolution's weight norm, the feature magnitude reaching it, and the gradient norms.

    <venv python> tools/diagnose_training.py --seed 20261011 --healthy-seed 20261010 --epochs 20
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_nrr as T  # noqa: E402


def trace(seed, dataset, epochs, channels, batch_size, learning_rate):
    """Trains `epochs` epochs and returns a per-epoch trace of the quantities that would explain a stall."""
    torch.manual_seed(seed)
    model = T.Upscaler(channels, 8, 8, 8, inputs=("color",))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    keys = list(dataset["train"].keys())
    count = dataset["sizes"]["train"]
    generator = torch.Generator().manual_seed(seed)
    rows = []
    for epoch in range(1, epochs + 1):
        order = torch.randperm(count, generator=generator)
        model.train()
        total, batches = 0.0, 0
        feature_magnitude = 0.0
        gradient_norm = 0.0
        for start in range(0, count, batch_size):
            batch = T.take(dataset["train"], order[start:start + batch_size], keys)
            optimizer.zero_grad()
            captured = {}
            handle = model.block2.register_forward_hook(
                lambda module, inputs, output: captured.update({"f": output.detach()}))
            prediction = model.residual(batch["color"], None, None, None)
            handle.remove()
            truth = batch["target"] - T.baseline_upscale(batch["color"])
            loss = torch.nn.functional.l1_loss(prediction, truth)
            loss.backward()
            feature_magnitude += float(captured["f"].abs().mean())
            gradient_norm += float(model.fusion.weight.grad.norm())
            optimizer.step()
            total += float(loss)
            batches += 1
        rows.append({"epoch": epoch, "loss": total / batches,
                     "output_weight_norm": float(model.output.weight.norm()),
                     "feature_abs_mean": feature_magnitude / batches,
                     "fusion_grad_norm": gradient_norm / batches})
    return rows


def main(argv):
    parser = argparse.ArgumentParser(description="Trace a stalling seed against a healthy one.")
    parser.add_argument("--data", default="models/training-data/godot-v1")
    parser.add_argument("--seed", type=int, required=True, help="the seed that stalls")
    parser.add_argument("--healthy-seed", type=int, default=0, help="a seed that trains, for contrast")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    args = parser.parse_args(argv[1:])

    _, dataset = T.load_dataset(args.data)
    seeds = [args.seed] + ([args.healthy_seed] if args.healthy_seed else [])
    for seed in seeds:
        label = "stalling" if seed == args.seed else "healthy"
        print("=== seed %d (%s) ===" % (seed, label))
        print("  epoch      loss   |W_out|   |features|   |grad fusion|")
        for row in trace(seed, dataset, args.epochs, args.channels, args.batch_size,
                         args.learning_rate):
            print("  %5d  %8.5f  %8.6f  %10.6f  %11.8f"
                  % (row["epoch"], row["loss"], row["output_weight_norm"], row["feature_abs_mean"],
                     row["fusion_grad_norm"]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))