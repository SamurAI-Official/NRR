"""stall_diag.py - why do some seeds never learn?

Five of eight stage-4 runs sat at exactly 0.01484 residual L1, which is the loss of a model that outputs no
residual at all - i.e. the output convolution never left its zero initialisation. This prints, per seed,
what the features entering that convolution look like and whether any gradient reaches the weights. Run:

    <venv python> tools/stall_diag.py
"""

import sys

import torch

sys.path.insert(0, "tools")
import train_nrr as T  # noqa: E402


def main():
    _, dataset = T.load_dataset("models/training-data/godot-v1")
    keys = list(dataset["train"].keys())
    batch = T.take(dataset["train"], slice(0, 16), keys)

    for seed in (20261001, 20261002):
        torch.manual_seed(seed)
        model = T.Upscaler(32, 8, 8, 8, inputs=("color",))
        captured = {}
        hook = model.block2.register_forward_hook(
            lambda module, inputs, output: captured.update({"f": output.detach()}))
        model.eval()
        with torch.no_grad():
            model(batch["color"], None, None, None)
        hook.remove()
        features = captured["f"]

        model.train()
        prediction = model.residual(batch["color"], None, None, None)
        truth = batch["target"] - T.baseline_upscale(batch["color"])
        loss = torch.nn.functional.l1_loss(prediction, truth)
        loss.backward()

        print("seed %d:" % seed)
        print("  features into output conv: min %.8f max %.8f fraction>0 %.4f"
              % (float(features.min()), float(features.max()),
                 float((features > 0).float().mean())))
        print("  grad norms: output.weight %.10f  fusion.weight %.10f  feature.weight %.10f"
              % (float(model.output.weight.grad.norm()), float(model.fusion.weight.grad.norm()),
                 float(model.feature[0].weight.grad.norm())))
        print("  output.weight abs max after init: %.10f" % float(model.output.weight.abs().max()))
        print("  loss %.6f" % float(loss))


if __name__ == "__main__":
    main()
