"""Architecture-diagram placeholder. Real version drawn manually in Inkscape."""

import matplotlib.pyplot as plt

from scripts.figures._common import save


def main() -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.text(
        0.5,
        0.5,
        "Multi-agent architecture diagram\n(Extractor × 3 → Critic → Consolidator)\n\nplaceholder — final version drawn in Inkscape",
        ha="center",
        va="center",
        fontsize=14,
    )
    ax.axis("off")
    save(fig, "fig_architecture")


if __name__ == "__main__":
    main()
