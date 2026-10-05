from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw"


def exact_signflip_p(values):
    values = np.asarray(values, dtype=float)
    observed = abs(values.mean())
    signs = np.asarray(list(product([-1.0, 1.0], repeat=len(values))))
    null_means = np.abs((signs * values).mean(axis=1))
    return float(np.mean(null_means >= observed - 1e-15))


def bootstrap_mean_ci(values, seed, n_boot=20000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(
        values,
        size=(n_boot, len(values)),
        replace=True,
    ).mean(axis=1)
    return tuple(np.percentile(draws, [2.5, 97.5]))


def holm_adjust(p_values):
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    running = 0.0
    m = len(p_values)

    for rank, idx in enumerate(order):
        candidate = (m - rank) * p_values[idx]
        running = max(running, candidate)
        adjusted[idx] = min(1.0, running)

    return adjusted


def analytical_random_baseline():
    """Recompute the manuscript uniform-random ranking baseline."""
    baseline = pd.read_csv(
        RAW / "battery_lopo_baseline_results.csv"
    )

    counts = (
        baseline[["battery_id", "battery", "n_transitions"]]
        .drop_duplicates()
        .sort_values("battery_id")
    )

    rows = []

    for _, row in counts.iterrows():
        n_states = int(row["n_transitions"])
        candidate_sizes = np.arange(1, n_states + 1, dtype=float)
        harmonic = np.cumsum(1.0 / candidate_sizes)

        rows.append(
            {
                "battery": row["battery"],
                "hits1": np.mean(1.0 / candidate_sizes),
                "hits3": np.mean(
                    np.minimum(3.0, candidate_sizes)
                    / candidate_sizes
                ),
                "mrr": np.mean(
                    harmonic / candidate_sizes
                ),
            }
        )

    product_values = pd.DataFrame(rows)

    print("\nANALYTICAL UNIFORM-RANDOM BASELINE")
    print(
        "Hits@1={:.6f}, Hits@3={:.6f}, MRR={:.6f}".format(
            product_values["hits1"].mean(),
            product_values["hits3"].mean(),
            product_values["mrr"].mean(),
        )
    )


def capacity_control():
    """Recompute the manuscript capacity-control comparison."""
    capacity_fold = pd.read_csv(
        RAW / "topology_ablation_fold_mean.csv"
    )

    matched = (
        capacity_fold[
            capacity_fold["model"] == "MatchedMLP"
        ]
        .set_index("battery")[["hits1", "hits3", "mrr"]]
    )

    mechanism_fold = pd.read_csv(
        RAW / "final_topology_mechanism_fold_mean.csv"
    )

    real = mechanism_fold.set_index("battery")

    if set(matched.index) != set(real.index):
        raise ValueError(
            "Battery indices do not match between capacity and "
            "final-mechanism result files."
        )

    # Preserve the saved held-out-product order used when the manuscript
    # bootstrap intervals were generated. This keeps the finite Monte Carlo
    # bootstrap draw sequence exactly reproducible.
    real = real.loc[matched.index]

    print("\nCAPACITY CONTROL")

    for metric in ["hits1", "hits3", "mrr"]:
        d = real[f"real_{metric}"] - matched[metric]

        ci = bootstrap_mean_ci(
            d,
            seed=20261200,
            n_boot=20000,
        )

        print(
            f"{metric}: "
            f"mean diff={d.mean():.6f}, "
            f"wins={(d > 0).sum()}/10, "
            f"95% CI=({ci[0]:.6f}, {ci[1]:.6f}), "
            f"exact p={exact_signflip_p(d):.9f}"
        )


def mechanism_control():
    """Recompute the primary topology-mechanism comparisons."""
    fold = pd.read_csv(
        RAW / "final_topology_mechanism_fold_mean.csv"
    )

    comparisons = [
        (
            "Real - Permuted",
            "real",
            "shuffle",
            20261201,
        ),
        (
            "Real - No-edge",
            "real",
            "noedge",
            20261202,
        ),
        (
            "Permuted - No-edge",
            "shuffle",
            "noedge",
            20261203,
        ),
    ]

    print("\nTOPOLOGY MECHANISM")

    h1_ps = []

    for label, a, b, seed in comparisons:
        d = fold[f"{a}_hits1"] - fold[f"{b}_hits1"]
        p = exact_signflip_p(d)
        h1_ps.append(p)

        ci = bootstrap_mean_ci(
            d,
            seed=seed,
            n_boot=20000,
        )

        print(
            f"Hits@1 {label}: "
            f"mean diff={d.mean():.6f}, "
            f"wins={(d > 0).sum()}/10, "
            f"95% CI=({ci[0]:.6f}, {ci[1]:.6f}), "
            f"exact p={p:.9f}"
        )

    print("Holm-adjusted primary Hits@1 p-values:")

    for (label, *_), p_adj in zip(
        comparisons,
        holm_adjust(h1_ps),
    ):
        print(f"  {label}: {p_adj:.9f}")

    for metric in ["hits3", "mrr"]:
        for label, a, b, _ in comparisons:
            d = (
                fold[f"{a}_{metric}"]
                - fold[f"{b}_{metric}"]
            )

            print(
                f"{metric} {label}: "
                f"mean diff={d.mean():.6f}, "
                f"wins={(d > 0).sum()}/10, "
                f"exact p={exact_signflip_p(d):.9f}"
            )


def no_progress_sensitivity():
    """Recompute the no-progress robustness analysis."""
    fold = pd.read_csv(
        RAW / "no_progress_sensitivity_fold_mean.csv"
    )

    comparisons = [
        ("Real - Permuted", "real", "shuffle"),
        ("Real - No-edge", "real", "noedge"),
        ("Permuted - No-edge", "shuffle", "noedge"),
        ("Real - MatchedMLP", "real", "matched"),
    ]

    bootstrap_seeds = {
        ("hits1", "Real - Permuted"): 20261201,
        ("hits1", "Real - No-edge"): 20261202,
        ("hits1", "Permuted - No-edge"): 20261203,
        ("hits1", "Real - MatchedMLP"): 20261204,
        ("hits3", "Real - Permuted"): 20261211,
        ("hits3", "Real - No-edge"): 20261212,
        ("hits3", "Permuted - No-edge"): 20261213,
        ("hits3", "Real - MatchedMLP"): 20261214,
        ("mrr", "Real - Permuted"): 20261221,
        ("mrr", "Real - No-edge"): 20261222,
        ("mrr", "Permuted - No-edge"): 20261223,
        ("mrr", "Real - MatchedMLP"): 20261224,
    }

    print("\nNO-PROGRESS SENSITIVITY")

    mean_rows = [
        (
            "RealGNN",
            fold["real_hits1"].mean(),
            fold["real_hits3"].mean(),
            fold["real_mrr"].mean(),
        ),
        (
            "FivePermutationMean",
            fold["shuffle_hits1"].mean(),
            fold["shuffle_hits3"].mean(),
            fold["shuffle_mrr"].mean(),
        ),
        (
            "NoEdgesGNN",
            fold["noedge_hits1"].mean(),
            fold["noedge_hits3"].mean(),
            fold["noedge_mrr"].mean(),
        ),
        (
            "MatchedMLP",
            fold["matched_hits1"].mean(),
            fold["matched_hits3"].mean(),
            fold["matched_mrr"].mean(),
        ),
    ]

    for label, h1, h3, mrr in mean_rows:
        print(
            f"{label}: Hits@1={h1:.6f}, "
            f"Hits@3={h3:.6f}, MRR={mrr:.6f}"
        )

    h1_topology_ps = []
    h1_topology_labels = []

    for metric in ["hits1", "hits3", "mrr"]:
        for label, a, b in comparisons:
            d = (
                fold[f"{a}_{metric}"]
                - fold[f"{b}_{metric}"]
            )
            p = exact_signflip_p(d)
            ci = bootstrap_mean_ci(
                d,
                seed=bootstrap_seeds[(metric, label)],
                n_boot=20000,
            )

            print(
                f"{metric} {label}: "
                f"mean diff={d.mean():.6f}, "
                f"wins={(d > 0).sum()}/10, "
                f"95% CI=({ci[0]:.6f}, {ci[1]:.6f}), "
                f"exact p={p:.9f}"
            )

            if metric == "hits1" and label != "Real - MatchedMLP":
                h1_topology_labels.append(label)
                h1_topology_ps.append(p)

    print("Holm-adjusted no-progress Hits@1 topology p-values:")

    for label, p_adj in zip(
        h1_topology_labels,
        holm_adjust(h1_topology_ps),
    ):
        print(f"  {label}: {p_adj:.9f}")


if __name__ == "__main__":
    analytical_random_baseline()
    capacity_control()
    mechanism_control()
    no_progress_sensitivity()
