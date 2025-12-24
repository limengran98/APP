# APP: Adaptive Prototype-guided Personalized Propagation

Official implementation of **"Adaptive Prototype-guided Personalized Propagation for Heterophilic Graphs with Missing Data"**.
APP is a unified framework to jointly address **heterophily** and **missing node features** in Graph Neural Networks, consisting of:

* **SRP** – *Semantic Rectification via Prototypes* for heterophily-aware propagation.
* **PVP** – *Personalized Virtual Propagation* for robust feature imputation.
* **ARS** – *Adaptive Representation Synergy* for prototype-guided fusion.

---

## 📌 Requirements & Installation

We recommend using a **Conda** environment to manage dependencies and avoid version conflicts.

### 1\. Create and Activate Environment

```bash
conda create -n app_env python=3.11
conda activate app_env
```

### 2\. Install PyTorch

Install the version compatible with your system (PyTorch \>= 2.1.0 recommended).

  * **For CUDA 11.8 (Example):**
    ```bash
    pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
    ```
  * **For CPU-only:**
    ```bash
    pip install torch torchvision torchaudio
    ```

### 3\. Install PyG Dependencies (Critical Step)

**Note:** `torch-scatter` and `torch-sparse` depend heavily on the specific PyTorch and CUDA versions. Installing them directly via pip may cause compilation errors. Please use the pre-built wheels:

Replace `${TORCH}` and `${CUDA}` with your installed versions. For example, for **PyTorch 2.1.0** and **CUDA 11.8**:

```bash
pip install torch_geometric
pip install torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
```

  * **Reference for common versions:**
      * PyTorch 2.1.0 + CUDA 11.8: `-f https://data.pyg.org/whl/torch-2.1.0+cu118.html`
      * PyTorch 2.1.0 + CPU: `-f https://data.pyg.org/whl/torch-2.1.0+cpu.html`

### 4\. Install Remaining Dependencies

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
* `arxiv`
* `genius`

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


