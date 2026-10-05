# Reproducibility guide

## 1. Experimental unit and split

The manuscript uses strict leave-one-product-out (LOPO) evaluation over 10 complete battery products. In each fold, one battery is held out in its entirety and the other nine products are used for model fitting. The held-out battery product is the independent unit for statistical inference.

Battery identity, raw component identifiers, raw fixation identifiers, and raw operation identifiers are excluded from the predictive representation. Categorical vocabularies are constructed from the training products only.

## 2. Notebook sequence

Run the notebooks in the following conceptual order.

### A. Initial cross-product baseline

`notebooks/Battery_Disassembly_MLP_vs_GNN_LOPO.ipynb`

Compares the attribute-only MLP with the real-topology GNN under the same 10 LOPO folds and three neural-training seeds.

### B. Capacity and preliminary topology ablation

`notebooks/Battery_Disassembly_Topology_Ablation_LOPO_v2.ipynb`

Adds the capacity-matched MLP and the preliminary topology control used to test whether parameter count alone explains the GNN advantage.

### C. Final topology-mechanism experiment

`notebooks/Battery_Disassembly_Final_Topology_Mechanism_LOPO.ipynb`

Compares `RealGNN`, `NoEdgesGNN`, and five `ShuffledGNN` realizations.

The historical implementation labels `shuffle` and `ShuffledGNN` correspond to the manuscript's **semantic-topology permutation control**. This is not unconstrained edge rewiring. A battery-specific node bijection is applied simultaneously to both endpoints of every recorded fixation edge, preserving the complete unlabeled graph up to node relabeling. Semantic node attributes and operation-to-node references remain at their original indices.

### D. No-progress sensitivity analysis

`notebooks/Battery_Disassembly_No_Progress_Sensitivity_LOPO.ipynb`

Repeats the key mechanism and capacity comparisons after removing normalized sequence progress `p_t=t/(T_b-1)` from the pairwise scoring head. It uses the same 10 LOPO folds, neural-training seeds, topology permutations, optimization settings, and negative-sampling protocol. The capacity-matched MLP width is reselected under the modified architecture and remains 88 in all folds.

## 3. Fixed randomization

Neural-training seeds: `17, 29, 43`.

Topology-permutation IDs: `1, 2, 3, 4, 5`.

For permutation ID `r` and battery ID `b`, the permutation generator seed is:

`20261119 + 10007*r + b`

The topology permutation is independent of the neural-training seed and the same battery/permutation assignment is reused across all three neural-training seeds.

## 4. Representation dimensions

Each operation uses five learned 12-dimensional categorical embeddings (skill, fixation class, source-part class, target-part class, and removed-part class), a four-dimensional operation-kind embedding, and three normalized local-degree features. The resulting operation input dimension is therefore `5×12 + 4 + 3 = 67`.

The primary pairwise scorer concatenates the current representation, candidate representation, absolute difference, elementwise product, and scalar normalized sequence progress. With a 64-dimensional operation representation, the primary pair representation has dimension `4×64 + 1 = 257`.

The no-progress sensitivity removes only the scalar progress input, so the corresponding pair representation has dimension `4×64 = 256`.

## 5. Graph controls

### Real topology

Recorded physical fixations are represented as typed undirected message-passing edges by making each physical fixation available in both directions.

### Permuted topology

The full physical graph is relabeled through a node permutation. The graph remains isomorphic to the recorded physical graph; edge types travel with transformed edges. Semantic component attributes and operation-to-node references are not permuted.

### No-edge

Cross-node edges are removed while retaining the graph architecture and trainable parameters. The graph-branch degree input is zeroed.

Operation-local normalized degree features derived from the recorded physical graph remain unchanged across all model and graph-control conditions. The mechanism conclusion therefore concerns the additional contribution of adjacency as a message-passing topology beyond this fixed local structural summary.

## 6. Training protocol

- 30 epochs
- AdamW
- learning rate `2e-3`
- weight decay `1e-4`
- dropout `0.10`
- 8 sampled negatives per training state
- neural seeds `[17, 29, 43]`
- one optimizer update per training battery per epoch
- 9 training products × 30 epochs = 270 optimizer updates per neural run
- no early stopping or test-set-based model selection

The final decision position is excluded from pairwise training because there is no later operation to sample as a negative.

## 7. Evaluation

At each held-out decision state, all remaining operations in the recorded reference sequence are scored. Metrics are Hits@1, Hits@3, and mean reciprocal rank (MRR). Ties within `1e-8` are handled using expected performance under uniform random ordering within the tied group.

The terminal state contains one candidate and therefore contributes a deterministic value of 1 to all three ranking metrics. The manuscript includes a sensitivity analysis excluding this state.

### Analytical uniform-random baseline

For a state with `n` candidates, uniform random ranking has expected Hits@1 `1/n`, Hits@3 `min(3,n)/n`, and MRR `H_n/n`, where `H_n` is the `n`th harmonic number. These values are averaged across states within each held-out product and then equally across the 10 products, matching the neural-model aggregation. The resulting overall expectations are:

- Hits@1: `0.029332`
- Hits@3: `0.075379`
- MRR: `0.089631`

The script `scripts/recompute_statistics.py` derives these values from the saved per-product transition counts.

## 8. Statistical analysis

Repeated neural-training seeds are averaged within held-out product. For the permuted-topology condition, the five permutations are also averaged within product before product-level comparisons.

The primary endpoint is Hits@1. Product-level uncertainty is summarized with 20,000 percentile-bootstrap replicates. Inference uses the exact two-sided paired sign-flip test over all `2^10 = 1024` sign assignments. Holm adjustment is applied to the three primary Hits@1 topology contrasts. Hits@3 and MRR are secondary supporting endpoints.

The final mechanism notebook uses bootstrap seeds `20261201`, `20261202`, and `20261203` for the three primary Hits@1 topology contrasts.

The no-progress sensitivity analysis is treated as a robustness analysis rather than a new confirmatory family. Its product-level statistics are provided separately in `results/no_progress_sensitivity_paired.csv`.

## 9. No-progress sensitivity result

Without normalized sequence progress, mean Hits@1 is approximately `0.293` for the real-topology GNN, `0.329` for the five-permutation mean, `0.207` for the no-edge GNN, and `0.193` for the capacity-matched MLP.

For Hits@1, the product-level differences are approximately:

- Real − Permuted: `-0.036`, exact `p=0.0547`
- Real − No-edge: `+0.087`, exact `p=0.00195`
- Permuted − No-edge: `+0.123`, exact `p=0.00195`
- Real − Capacity-matched MLP: `+0.100`, exact `p=0.00195`

The qualitative mechanism conclusion therefore survives removal of explicit sequence progress.

## 10. Included outputs

`results/raw/` contains seed-level and product-level CSV outputs from the completed neural experiments, including the no-progress sensitivity run. Seed-level rows and permutation rows are repeated measurements and must not be treated as independent products.

`results/paper_paired_statistics.csv` contains the rounded primary manuscript-level statistics.

`results/no_progress_sensitivity_paired.csv` contains the product-level no-progress sensitivity contrasts and uncertainty summaries.

`results/random_baseline.csv` contains the analytical uniform-random baseline reported in the manuscript.

## 11. Terminology note

Some implementation names retain `shuffle` and `ShuffledGNN` for backward compatibility with the completed experiment outputs. The manuscript uses the more precise terms **permuted topology**, **permuted-topology GNN**, and **semantic-topology permutation control**.
