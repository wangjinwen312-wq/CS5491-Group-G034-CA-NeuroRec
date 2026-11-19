# Content-Aware NeuroRec

## 1. Project Overview

Content-Aware NeuroRec is a Python / PyTorch movie recommendation project built on the MovieLens 1M dataset. It investigates whether adding movie genre information to Neural Collaborative Filtering (NeuralCF) improves rating prediction and Top-K recommendation compared with traditional collaborative filtering. We compare five methods under the same data split and evaluation: UserCF, ItemCF, MF, NeuralCF, and the proposed genre-aware ContentAware model, plus a GenreOff ablation that disables the genre branch. Models are scored by RMSE / MAE (rating prediction) and Precision@K / Recall@K / NDCG@K (Top-K), with extra results for sparse-history users and movies, averaged over five random seeds.

## 2. Environment Setup

### 2.1 Requirements

The project has been developed and tested with the following environment:

- **Operating System**: Windows, macOS, or Linux.
- **Python**: 64-bit Python 3.12.
- **Deep Learning Framework**: PyTorch (CPU build).
- **Hardware**: CPU is sufficient; no dedicated GPU is required.
- **Dataset**: MovieLens 1M.

Two dependency files are provided:

- `requirements.txt`: Specifies compatible dependency version ranges.
- `requirements-lock.txt`: Contains pinned dependency versions used for the verified Windows / Python 3.12 CPU environment.

If Python is not installed, download it from the official website:
<https://www.python.org/downloads/>

On Windows, make sure to enable "Add Python to PATH" and "Python Launcher" during installation.

### 2.2 Installation

Open a terminal in the repository root directory.

**Windows (PowerShell):**

Create a virtual environment:

```powershell
py -3.12 -m venv .venv
```

Install the dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
```

**macOS / Linux:**

Create a virtual environment:

```bash
python3.12 -m venv .venv
```

Install the dependencies:

```bash
./.venv/bin/python -m pip install -r requirements-lock.txt
```

## 3. Reproducing the Experiments

All commands below are run from the repository root directory. The default configuration is `configs/full.json`; `configs/quick.json` is used only for debugging. On macOS / Linux, replace the Windows interpreter `.\.venv\Scripts\python.exe` with `./.venv/bin/python` in every command.

First, prepare the data without training any models:

```powershell
.\.venv\Scripts\python.exe main.py --prepare
./.venv/bin/python main.py --prepare
```

This downloads and verifies the official MovieLens 1M dataset, extracts and cleans the rating and movie information, builds user and movie ID mappings, and generates the time-based training, validation, and test splits. Processed dataset files appear under `data/`, and dataset statistics and split information are printed in the terminal. The step typically takes less than one minute, excluding download time.

Before the full run, you may check the pipeline with the quick debug mode:

```powershell
.\.venv\Scripts\python.exe main.py --quick
./.venv/bin/python main.py --quick
```

It uses one random seed and trains neural models for at most two epochs to verify data loading, model initialization and training, evaluation metrics, result generation, and plotting. Debug results and figures are written to `results/quick/` with metrics printed in the terminal, usually within a few minutes on a CPU. These results are for debugging only and must not replace the official experimental results.

To reproduce the results in the final report, run the official experiment:

```powershell
.\.venv\Scripts\python.exe main.py
./.venv/bin/python main.py
```

The five algorithms are trained and evaluated independently over the five seeds 11, 22, 33, 44, and 55, and the GenreOff ablation model is retrained five times independently, for a total of 30 actual model fits; neural models train for at most 12 epochs with early stopping based on validation performance (patience 3), and all official hyperparameters are fixed in the configuration files. All official results are saved under `results/full/`: the automatically generated `RESULTS.md`, eight figure types in PNG and PDF under `figures/`, per-run `runs.csv`, cross-seed mean and sample standard deviation in `summary.csv`, genre ablation results in `ablation*.csv`, detailed execution records in `execution_log.csv`, and saved model weights in `checkpoints/`. The reported full experiment took 379.33 seconds (about 6.3 minutes) on a CPU with 4 threads. This includes data loading, 30 model fits, evaluation, and figure generation. The total elapsed time is recorded as `elapsed_seconds` in `results/full/status.json`. Per-fit timings are recorded in `results/full/execution_log.csv`. Runtime varies with hardware.

Once the official result CSVs exist, figures can be regenerated without retraining:

```powershell
.\.venv\Scripts\python.exe main.py --plot-only
./.venv/bin/python main.py --plot-only
```

This reads the existing results and refreshes the PNG and PDF figures under `results/full/figures/`, usually in less than one minute. Finally, the 47 automated checks over data processing, model implementations, training, metrics, and result integrity can be run as follows; the terminal lists each check and ends with `OK` when all pass, and the suite finishes in under one minute using small constructed data that does not require downloading the full MovieLens dataset.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
./.venv/bin/python -m unittest discover -s tests -v
```

## 4. Recommendation Demo

After the official experiments finish, the trained ContentAware model can generate personalized recommendations for a specific user. Run the command below; the example uses a verified training-set user (6040):

```powershell
.\.venv\Scripts\python.exe recommend.py --user-id 6040 --top-k 10
./.venv/bin/python recommend.py --user-id 6040 --top-k 10
```

The `--user-id` argument takes the original MovieLens user ID, `--top-k` sets the number of recommendations (default 10), and `--results` points to the results directory (default `results/full`). The program loads the trained model and prints a ranked list showing the recommendation position, movie title, movie genres, and predicted rating; this usually takes a few seconds on a CPU.

Note that the selected user must have appeared in the training set. Although MovieLens user IDs range from 1 to 6040, the time-based 80/10/10 split means some users have all their ratings in the validation or test sets and no training ratings, and such IDs are rejected with an error; pick a user recorded in the training-set statistics or use a verified ID such as 6040. The demo uses the checkpoint saved from the first official random seed, excludes movies the user has already rated across the complete dataset, and is kept separate from the official Top-K evaluation procedure.

## 5. Data

### 5.1 Dataset Source

The project uses the official MovieLens 1M dataset provided by GroupLens, which contains approximately one million real user ratings and movie metadata. The program reads `ratings.dat` (user IDs, movie IDs, ratings, and timestamps), `movies.dat` (movie IDs, titles, and genre information), and `users.dat` (user demographic information, used for reference only); ratings range from 1 to 5. The official dataset page is <https://grouplens.org/datasets/movielens/1m/>, and the archive can be downloaded from <https://files.grouplens.org/datasets/movielens/ml-1m.zip>.

### 5.2 Data Preparation

The dataset is downloaded and processed automatically during the first run. If automatic downloading fails, manually download `ml-1m.zip` from the GroupLens website, create a `data/` directory in the repository root, place the archive inside it, and run the data preparation or experiment command again; the program verifies and extracts the archive before processing.

## 6. Experimental Protocol

The complete experimental methodology is documented in [EXPERIMENT_PROTOCOL_EN.md](EXPERIMENT_PROTOCOL_EN.md).