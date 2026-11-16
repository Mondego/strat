"""Torus geometry shared by STRAT and the Transformer."""

import torch
import torch.nn as nn


def torus_offsets(height: int, width: int):
    index = torch.arange(height * width)
    row, col = index // width, index % width
    dr = ((row[None, :] - row[:, None]) % height).long()
    dc = ((col[None, :] - col[:, None]) % width).long()
    return dr, dc


class D4RelativeBias(nn.Module):
    """Learned relative-position bias tied across rotations and reflections."""

    def __init__(self, n_heads: int, height: int, width: int):
        super().__init__()
        self.table = nn.Parameter(torch.zeros(n_heads, height, width))

        dr, dc = torus_offsets(height, width)
        transforms = [
            (dr, dc),
            ((-dc) % height, dr),
            ((-dr) % height, (-dc) % width),
            (dc, (-dr) % width),
            ((-dr) % height, dc),
            (dr, (-dc) % width),
            (dc, dr),
            ((-dc) % height, (-dr) % width),
        ]
        for index, (tr, tc) in enumerate(transforms):
            self.register_buffer(f"dr_{index}", tr)
            self.register_buffer(f"dc_{index}", tc)

    def forward(self):
        tied = [
            self.table[:, getattr(self, f"dr_{index}"), getattr(self, f"dc_{index}")]
            for index in range(8)
        ]
        return torch.stack(tied).mean(dim=0)
