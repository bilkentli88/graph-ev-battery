# Graph Inductive Bias Audit for EV Battery Disassembly

Reproducibility materials for the manuscript **“Auditing Graph Inductive Bias in EV Battery Disassembly: Leave-One-Product-Out Evaluation with Semantic–Topology Permutation Controls.”**

## Research question

This study asks whether the recorded physical part/fixation topology provides transferable predictive information for reference next-operation ranking on completely unseen EV battery products, beyond product-invariant operation attributes, trainable parameter count, and generic cross-node message passing.

The experimental design uses strict leave-one-product-out (LOPO) evaluation over 10 battery products and separates three possible explanations for a GNN advantage: increased parameter count, generic cross-node message passing, and correct semantic–topological alignment with the recorded physical fixation graph.

## Main model conditions

- Attribute-only MLP
- Capacity-matched MLP
- Real-topology GNN
- No-edge GNN
- Five semantic–topology permutation controls

A separate sensitivity experiment removes normalized sequence progress from the pairwise scoring head and repeats the key topology and capacity comparisons.

## Important terminology

The original experiment notebooks use the historical implementation labels `ShuffledGNN` and `graph_mode="shuffle"`. In the manuscript these are called **permuted-topology GNNs** or **semantic–topology permutation controls**.

This intervention is **not unconstrained edge rewiring**. For each permutation realization, a battery-specific node bijection is applied simultaneously to both endpoints of every physical fixation edge. The complete unlabeled graph therefore remains graph-isomorphic to the recorded physical graph, while semantic node attributes and operation-to-node references remain at their original indices. The intervention breaks semantic component ↔ structural-position alignment.

The five permutation realizations are distinct pseudorandom controls. They are repeated controls within the same held-out product and are not treated as independent experimental units.

## Repository structure

```text
.
├── README.md
├── LICENSE
├── requirements.txt
├── configs/
│   └── experiment_config.yaml
├── data/
│   └── README.md
├── docs/
│   └── reproducibility.md
├── manuscript/
│   └── code_data_availability.tex
├── notebooks/
│   ├── Battery_Disassembly_MLP_vs_GNN_LOPO.ipynb
│   ├── Battery_Disassembly_Topology_Ablation_LOPO_v2.ipynb
│   ├── Battery_Disassembly_Final_Topology_Mechanism_LOPO.ipynb
│   └── Battery_Disassembly_No_Progress_Sensitivity_LOPO.ipynb
├── results/
│   ├── README.md
│   ├── individual_permutations.csv
│   ├── paper_paired_statistics.csv
│   ├── no_progress_sensitivity_paired.csv
│   ├── random_baseline.csv
│   ├── recomputed_statistics.txt
│   ├── singleton_sensitivity.csv
│   ├── structural_associations.csv
│   └── raw/
│       └── experiment CSV outputs
└── scripts/
    └── recompute_statistics.py
```

## Dataset

The experiments use:

**Baucks, M. and Morasch, A. (2026). Knowledge Graph Dataset of Product Structures and Disassembly Sequences of 10 EV Battery Packs. Karlsruhe Institute of Technology.**

DOI: https://doi.org/10.35097/emz24pksshndq468

The dataset is not redistributed in this repository. Download the archive from the DOI landing page and follow `data/README.md`.

## Environment

The notebooks use Python, NumPy, pandas, and PyTorch and were developed in a Google Colab-style environment. Install the required packages with:

```bash
pip install -r requirements.txt
```

A GPU is recommended for rerunning the full neural experiments, although the notebooks can fall back to CPU.

## Fixed experiment configuration

- LOPO folds: 10
- Neural-training seeds: `[17, 29, 43]`
- Semantic–topology permutation IDs: `[1, 2, 3, 4, 5]`
- Epochs: `30`
- Negative samples per training state: `8`
- Optimizer: AdamW
- Learning rate: `2e-3`
- Weight decay: `1e-4`
- Dropout: `0.10`
- Categorical embedding dimension: `12`
- Operation input dimension: `67`
- Standard operation hidden dimension: `64`
- Capacity-matched MLP width: `88`
- GNN hidden dimension: `32`
- Edge-type embedding dimension: `8`
- Message-passing layers: `2`
- Primary pair representation dimension: `257`
- No-progress pair representation dimension: `256`
- Topology-permutation base constant: `20261119`

For permutation ID `r` and battery ID `b`, the topology-permutation generator uses:

```text
20261119 + 10007 * r + b
```

The topology permutation is independent of the neural-training seed and the same battery/permutation assignment is reused across the three neural-training seeds.

## Evaluation baselines and sensitivity analyses

The manuscript reports an analytical uniform-random ranking baseline. For a decision state with `n` candidates, the expected values are `1/n` for Hits@1, `min(3,n)/n` for Hits@3, and `H_n/n` for MRR, where `H_n` is the `n`th harmonic number. Using the same within-product and across-product averaging as the neural models gives Hits@1 `0.029`, Hits@3 `0.075`, and MRR `0.090`.

Two robustness checks are included:

- removal of the deterministic terminal singleton state;
- removal of normalized sequence progress from the pairwise scoring head.

The no-progress sensitivity run preserves the main qualitative mechanism pattern: real and permuted graph conditions remain above the no-edge control, while the real topology does not demonstrate an additional advantage over the permutation mean under the exact paired product-level test.

## Principal findings

At the held-out-product level, the real-topology GNN exceeds the capacity-matched MLP on all 10 products. Both real and permuted graph conditions also exceed the no-edge control on all 10 products. In contrast, the recorded physical topology does not demonstrate an advantage over the five-permutation mean under the exact two-sided product-level sign-flip test.

The independent statistical unit is the held-out battery product (`n=10`). Training seeds and topology permutations are repeated measurements and are not counted as independent observations.

See `results/paper_paired_statistics.csv` for the primary manuscript-level statistics, `results/no_progress_sensitivity_paired.csv` for the no-progress sensitivity statistics, and `results/raw/` for experiment outputs.

## Recompute product-level statistics

```bash
python scripts/recompute_statistics.py
```

The script recomputes the analytical random baseline, capacity-control statistics, topology-mechanism contrasts, no-progress sensitivity contrasts, exact paired sign-flip tests, product-level bootstrap intervals, and Holm-adjusted primary Hits@1 topology p-values from the saved CSV files.

## Scope of the topology conclusion

Operation-local normalized degree features derived from the recorded physical graph are retained across the model conditions. The topology intervention therefore evaluates the **additional contribution of physical adjacency as a message-passing topology beyond this fixed local structural summary**. It should not be interpreted as showing that all information derived from physical connectivity is uninformative.

## License

Repository code is released under the MIT License. The external dataset remains subject to the terms provided by its original publisher.
