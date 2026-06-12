# Reproducing paper experiments

Below, we outline the steps to reproduce the experimental results presented in
our paper, "Continuous SUN (Stable, Unique, and Novel) Metric for Generative
Modeling of Inorganic Crystals," submitted to IOP Machine Learning: Science and
Technology.

1. **Set up the environment**

    ```bash
    git clone https://github.com/WMD-group/xtalmet.git -b iop-mlst
    cd xtalmet
    uv sync --all-extras
    ```
    (Optional) Run pytests to make sure that the setup is successful. (Some tests may take hours to run.)

    ```bash
    uv run pytest tests/
    ```

2. **Evaluate crystal samples from generative models**

    For each model, run the evaluation script:

    ```bash
    uv run python examples/journal/eval.py --model MODEL_NAME
    ```

    where `MODEL_NAME` is one of `adit`, `cdvae`, `chemeleon`, `chemeleon2`, `diffcsp`, `diffcsppp`, `mattergen`, `test`, `chemeleon2_rl_bsun`, `chemeleon2_rl_csun`, and `chemeleon2_rl_csun_u10`.
    Depending on the capabilities of your machine, each command may take hours or even a whole day to complete.
    So, we provide pre-computed evaluation results on [Hugging Face](https://huggingface.co/datasets/masahiro-negishi/xtalmet/tree/main/journal/results/mp20) under `journal/results/mp20/`. 
    To use them, download and place them at `examples/journal/results/mp20/` before running `visualize.ipynb`.
    The samples generated from these models are also available on [Hugging Face](https://huggingface.co/datasets/masahiro-negishi/xtalmet) under `mp20/model`.
    If you would like to sample from the models by yourself, follow the instructions in their original repositories.
    To try out reinforcement learning on Chemeleon2, please see the instructions [here](https://hspark1212.github.io/chemeleon2/index-2/).
    The reinforcement-learning trajectories are available as [`rl_trajectory.csv`](https://huggingface.co/datasets/masahiro-negishi/xtalmet/blob/main/journal/results/rl_trajectory.csv), which `visualize.ipynb` downloads automatically.

3. **Visualize the results**

    Once you have obtained the experimental results, run
    `examples/journal/visualize.ipynb` to generate the figures and tables
    presented in the paper. The preprocessing may take roughly 30 minutes.

## Reproducing revision analyses

The notebooks and scripts under `examples/journal/revision/` reproduce the
additional analyses prepared during peer review. They use the base evaluation
caches under `examples/journal/results/mp20/` and write additional caches under
`examples/journal/revision/preprocess/`.

1. **Prepare the base caches**

    Run `examples/journal/eval.py` as described above for `adit`, `cdvae`,
    `chemeleon`, `chemeleon2`, `diffcsp`, `diffcsppp`, `mattergen`, and `test`,
    or download the pre-computed results. Each model directory must contain the
    energy-above-hull values, embeddings, and uniqueness and novelty matrices
    used by the revision analyses.

    The revision notebook also reads these files produced by the preprocessing
    cells in `examples/journal/visualize.ipynb`:

    ```text
    examples/journal/preprocess/train_samples.pkl.gz
    examples/journal/preprocess/uni.csv
    examples/journal/preprocess/nov.csv
    ```

    The original notebook computes `uni.csv` and `nov.csv` for all base and
    reinforcement-learning models. Therefore, either retain the provided
    preprocessing files or prepare evaluation caches for every model listed in
    step 2 before regenerating them.

2. **Compute AMD neighbor-count caches**

    The default AMD matrices use `k=100` and are created by `eval.py`. Generate
    the additional `k=25`, `k=50`, and `k=200` embeddings and matrices with:

    ```bash
    bash examples/journal/revision/run_amd_k.sh
    ```

    This step requires `examples/journal/preprocess/train_samples.pkl.gz` for
    the novelty matrices.

3. **Compute model-pair distance caches**

    Generate the sampled `d_elmd+amd` matrices used for the set-distance
    analysis:

    ```bash
    uv run python examples/journal/revision/compute_model_pair_elmd_amd.py
    ```

    By default, this samples 1,000 structures per model with random seed 0 and
    writes the sampled indices and all unordered model-pair matrices under
    `examples/journal/revision/preprocess/model_pair_elmd_amd/n1000_seed0/`.
    It requires `emb_elmd+amd.pkl.gz` in each base model results directory.

4. **Run the revision notebook**

    Run `examples/journal/revision/visualize.ipynb`. It derives summary CSV
    files under `examples/journal/revision/preprocess/` and writes figures under
    `examples/journal/revision/figures/`.

    Cache generation is expensive. The scripts and notebook reuse existing
    files by default; remove the relevant cache or enable its overwrite option
    only when recomputation is required.
