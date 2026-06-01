"""Shared helpers for v2 figures: same palette, write to v2 figures dir."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

FIG_DIR = Path("outputs/paper_data_v2/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = sns.color_palette("colorblind")
sns.set_theme(context="paper", style="whitegrid", font_scale=1.1, palette="colorblind")


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())
