# Dataset

This repository does not redistribute the EV-battery dataset.

Download the dataset used in the manuscript from:

**Knowledge Graph Dataset of Product Structures and Disassembly Sequences of 10 EV Battery Packs**  
Baucks, Marina and Morasch, Alexander (2026), Karlsruhe Institute of Technology.  
DOI: https://doi.org/10.35097/emz24pksshndq468

The original experiment notebooks expect the archive name:

`10.35097-emz24pksshndq468.tar`

## Google Colab

Upload the archive to `/content`. The notebooks search `/content` recursively and extract the archive when needed.

## Local execution

Either adapt the notebook path-configuration cell so that `root` points to the directory containing the archive, or reproduce the Colab-style directory layout in your chosen working directory.

The notebooks read the extracted CSV files including `battery_catalogue.csv`, `parts.csv`, `fixations.csv`, and `operations.csv`.

Do not commit the raw dataset archive or extracted files to this repository.
