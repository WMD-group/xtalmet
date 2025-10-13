# Reproducing paper experiments
Below, we outline the steps to reproduce the experimental results in our paper, "Continuous Uniqueness and Novelty Metrics for Generative Modeling of Inorganic Crystals," presented at the NeurIPS 2025 AI4Mat workshop.

1. **Set up the environment**

    ```bash
    git clone https://github.com/WMD-group/xtalmet.git
    cd xtalmet
    uv sync
    ```
    (Optional) Run pytests to make sure that the setup is successful. (Some tests may take a while to run.)
    ```bash
    uv run pytest tests/
    ```

2. **Evaluate crystals samples from generative models**

    For each model, run the evaluation script:
    ```bash
    uv run python examples/workshop/eval.py --model MODEL_NAME --screen none --metric both
    ```
    ```bash
    uv run python examples/workshop/eval.py --model MODEL_NAME --screen ehull --metric both
    ```
    where `MODEL_NAME` is one of `adit`, `cdvae`, `chemeleon`, `diffcsp`, `diffcsppp`, and `mattergen`.
    Depending on the capabilities of your machine, each command may take hours or even a whole day to complete. 
    Therefore, we provide pre-computed results in the `results.zip` file. 
    To skip the above commands, simply unzip the file.

3. **Visualize the results**

    Once you have obtained the evaluation results, run the visualisation scripts in 'visualize.ipynb' to generate the figures and tables presented in the paper.
