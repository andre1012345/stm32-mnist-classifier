# HPC Intrusion Detection — Hybrid MPI + OpenMP Ensemble Classifier

> Parallel machine learning pipeline for network intrusion detection using the CICIDS2017 dataset.  
> Implements a heterogeneous ensemble (Random Forest + Logistic Regression + K-NN) with a hybrid MPI + OpenMP parallelisation strategy.

---

## Results Summary

| Configuration | Ranks | OMP Threads | Time | Speedup | Efficiency | F1 |
|---|---|---|---|---|---|---|
| Sequential baseline | 1 | 1 | 589.227s | 1.00x | 100.0% | 0.9930 |
| OpenMP only | 1 | 4 | 209.386s | 2.81x | 70.3% | 0.9930 |
| MPI + OpenMP | 3 | 2 | 196.367s | **3.00x** | **100.0%** | 0.9930 |
| MPI + OpenMP | 4 | 2 | 225.673s | 2.61x | 65.3% | 0.9899 |

**Optimal configuration: p=3, 2 OpenMP threads per rank (Sp=3.00x, E=100%)**

---

## Architecture

### Parallelisation Strategy (PCAM)

```
Rank 0  →  Random Forest training (OpenMP) + RF inference + Aggregator
Rank 2  →  Logistic Regression training + LR inference
Rank 1,3→  KNN workers — data-parallel reference set split (OpenMP)
```

### Two levels of parallelism

**Task parallelism (MPI):** RF, LR, and KNN execute concurrently on dedicated ranks.  
**Data parallelism (MPI + OpenMP):** KNN reference set split across worker ranks; inner distance loop parallelised with OpenMP threads within each rank.

### Ensemble

Weighted soft voting with empirically tuned weights and threshold:

```
RF weight=0.354  LR weight=0.302  KNN weight=0.343  τ=0.63
```

Weights derived from per-model validation F1 scores on the Python baseline.

---

## Project Structure

```
HPC-intrusion/
├── data/
│   └── raw/                    ← CICIDS2017 CSV files (not tracked)
├── src/                        ← Python ML pipeline
│   ├── config.py               ← paths, hyperparameters
│   ├── prepare_data.py         ← load, clean, normalise, stratified split
│   ├── train_evaluate.py       ← train models, ROC/PR curves, threshold tuning
│   ├── export_for_cpp.py       ← export numpy artifacts to binary for C++
│   └── artifacts/              ← generated numpy + binary files (not tracked)
├── src_cpp/                    ← C++ HPC implementation
│   ├── main.cpp                ← sequential baseline
│   ├── main_mpi.cpp            ← hybrid MPI + OpenMP
│   ├── common.h                ← types, timer, metrics
│   ├── data_io.h               ← binary file loading
│   ├── rf.h                    ← Random Forest (OpenMP parallel)
│   ├── lr.h                    ← Logistic Regression
│   ├── knn.h                   ← KNN inference (OpenMP parallel)
│   └── ensemble.h              ← weighted soft voting
└── results/                    ← plots and metrics (generated)
    ├── roc_curves.png
    ├── pr_curves.png
    ├── threshold_tuning.png
    └── metrics.json
```

---

## Requirements

### Python
```
python >= 3.10
numpy pandas scikit-learn matplotlib faiss-cpu
```

### C++
```
g++ >= 11  with C++17
OpenMPI >= 4.1
```

---

## How to Run

### Step 1 — Python ML Pipeline

```bash
cd src

# Install dependencies
pip install numpy pandas scikit-learn matplotlib faiss-cpu

# Edit DATA_DIR in config.py to point at your CICIDS2017 CSV folder

# 1. Clean, normalise, stratified split
python3 prepare_data.py

# 2. Train models, tune threshold, generate plots
python3 train_evaluate.py

# 3. Export binary files for C++
python3 export_for_cpp.py
```

### Step 2 — Sequential Baseline

```bash
cd src_cpp
g++ -O3 -std=c++17 -o intrusion main.cpp
./intrusion
```

### Step 3 — OpenMP Only

```bash
g++ -O3 -std=c++17 -fopenmp -o intrusion_omp main.cpp
OMP_NUM_THREADS=4 ./intrusion_omp
```

### Step 4 — Hybrid MPI + OpenMP

```bash
mpic++ -O3 -std=c++17 -fopenmp -o intrusion_mpi main_mpi.cpp

# p=3 — optimal configuration
OMP_NUM_THREADS=2 mpirun -np 3 ./intrusion_mpi

# p=4 — strong scaling
OMP_NUM_THREADS=2 mpirun -np 4 ./intrusion_mpi
```

---

## Dataset

**CICIDS2017** — Canadian Institute for Cybersecurity Intrusion Detection Dataset 2017.  
Available at: https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset

15 traffic classes across 8 capture days:

| Class | Samples (sampled) |
|---|---|
| BENIGN | 322,950 |
| DDoS | 28,356 |
| PortScan | 27,739 |
| DoS Hulk | 16,679 |
| FTP-Patator | 890 |
| SSH-Patator | 661 |
| Bot | 514 |
| Web Attacks | 639 |
| Other DoS | 1,556 |

**Sampling:** 50,000 rows per CSV file (stratified), 70/30 train/test split.  
**Normalisation:** StandardScaler fitted on training set only — no data leakage.

---

## Key Design Decisions

**Why all 8 days?**  
The original project used only Tuesday's capture. This misses DDoS, PortScan, Bot, and Web Attack classes entirely. Using all 8 days produces a model that generalises across the full threat landscape.

**Why weighted soft voting?**  
Individual model F1 scores differ significantly (RF=0.997, KNN=0.966, LR=0.850). Equal weighting would let the weakest model drag down the ensemble. Weights proportional to validation F1 ensure the strongest model dominates.

**Why τ=0.63 and not 0.4?**  
The threshold was chosen empirically by maximising F1 on the validation set across the range [0.1, 0.9]. The original hardcoded τ=0.4 was unjustified.

**Why p=3 is optimal?**  
RF training on Rank 0 is the sequential bottleneck (~143s). Adding more KNN workers (p=4) cannot reduce this — they finish in ~45s and wait. This is Amdahl's Law: sequential fraction s≈33% caps speedup at Sp_max=3.0x.

**Why KNN data parallelism introduces approximation?**  
Each worker finds k=15 nearest neighbours from its local chunk rather than the global reference set. Equal-weight averaging of partial probabilities is a principled approximation. The accuracy cost is F1: 0.9930 → 0.9899 at p=4.

---

## Hardware

| Component | Specification |
|---|---|
| System | HP EliteBook 830 G5 |
| CPU | Intel Core i5 8th Gen @ 1.70GHz |
| Cores | 4 physical / 8 logical (HT) |
| RAM | 16GB DDR4 |
| OS | Windows 11 / WSL2 Ubuntu 22.04 |
| MPI | OpenMPI 4.1.2 |
| Compiler | GCC/G++ 11.4.0 `-O3 -fopenmp` |

---

## Author

Andrea Fascì — High Performance Computing, Università degli Studi di Messina, 2025/2026  
GitHub: https://github.com/andre1012345/HPC-intrusion
