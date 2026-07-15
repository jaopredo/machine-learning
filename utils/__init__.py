from __future__ import annotations

import torch


def move_cache(value, device: torch.device):
	if isinstance(value, torch.Tensor):
		return value.to(device)
	if isinstance(value, list):
		return [move_cache(item, device) for item in value]
	if isinstance(value, tuple):
		return tuple(move_cache(item, device) for item in value)
	if isinstance(value, dict):
		return {key: move_cache(item, device) for key, item in value.items()}
	return value

