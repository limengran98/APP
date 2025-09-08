# APP: Adaptive Prototype-guided Personalized Propagation

Official implementation of **"Adaptive Prototype-guided Personalized Propagation for Heterophilic Graphs with Missing Data"**.
APP is a unified framework to jointly address **heterophily** and **missing node features** in Graph Neural Networks, consisting of:

* **SRP** – *Semantic Rectification via Prototypes* for heterophily-aware propagation.
* **PVP** – *Personalized Virtual Propagation* for robust feature imputation.
* **ARS** – *Adaptive Representation Synergy* for prototype-guided fusion.

---
### ⚠️ Due to the double-blind review policy, the complete dataset and trainable parameters will be released once the paper is published.  

## 📌 Requirements

```bash
python >= 3.11
numpy
scipy
scikit-learn
torch >= 2.1.1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 📂 Dataset

The code supports multiple graph datasets:

* `Roman-empire`
* `Amazon-ratings`
* `Minesweeper`
* `Tolokers`
* `Squirrel`

Ensure datasets are placed in the correct folder (default: `../data/`).

---

## 🚀 Usage Examples

Run APP on different datasets with tuned parameters:

```bash
# Roman-empire
python main.py --data Roman-empire --num_iter 1 --gamma 0.3 --mu 0.1

# Amazon-ratings
python main.py --data Amazon-ratings --num_iter 7 --gamma 0.9 --mu 5

# Minesweeper
python main.py --data Minesweeper --num_iter 1 --gamma 0.2 --mu 0.1

# Tolokers
python main.py --data Tolokers --num_iter 2 --gamma 0.9 --mu 0.1

# Squirrel
python main.py --data squirrel --num_iter 1 --gamma 0.9 --mu 0.1
```
---

## 📊 Output

If `--save_results` is enabled, results are stored in:

```
./result/{dataset}results.csv
```


