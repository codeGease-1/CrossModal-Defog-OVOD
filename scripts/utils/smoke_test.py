#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Real small-tensor forward/backward check; no dataset or checkpoint required.

Run: python scripts/utils/smoke_test.py
Uses base_channels=4 and small even-sized images to check the computation graph.
It does not replace a full-size experiment or data validation.
"""
import sys
from pathlib import Path


def main():
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))
    try:
        import torch
        from src.models.haze_density import HazeDensityNet
        from src.models.haze_density.physical_prior import PhysicalPriorModule, PRIOR_VERSION
    except ImportError as error:
        print(f"[FAIL] Missing runtime dependency: {error}")
        return False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42)
    print(f"torch={torch.__version__}, device={device}, prior={PRIOR_VERSION}")
    try:
        model = HazeDensityNet(base_channels=4).to(device)
        prior = PhysicalPriorModule(window_size=3, guided_radius=2).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        for shape in ((1, 3, 32, 48), (2, 3, 48, 32)):
            images = torch.rand(shape, device=device)
            with torch.no_grad():
                targets = prior(images)
            predictions = model(images)
            expected = (shape[0], 1, shape[2], shape[3])
            for name, tensor in (('target', targets), ('prediction', predictions)):
                if tuple(tensor.shape) != expected:
                    raise RuntimeError(f'{name} shape: {tuple(tensor.shape)} != {expected}')
                if not torch.isfinite(tensor).all():
                    raise RuntimeError(f'{name} contains NaN/Inf')
                if tensor.min() < 0 or tensor.max() > 1:
                    raise RuntimeError(f'{name} is outside [0, 1]')
            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(predictions, targets)
            loss.backward()
            gradients = [parameter.grad for parameter in model.parameters()
                         if parameter.grad is not None]
            if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
                raise RuntimeError('Missing or non-finite model gradients')
            optimizer.step()
            print(f'[OK] {shape} -> {expected}, loss={loss.item():.6f}, backward/step complete')
    except Exception as error:
        print(f'[FAIL] {type(error).__name__}: {error}')
        return False
    print('[OK] Small-tensor smoke check passed')
    return True


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
