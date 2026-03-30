# Reproducing paper experiments
Below, we outline the steps to reproduce the experimental results presented in our paper, "Continuous SUN (Stable, Unique, and Novel) Metric for Generative Modeling of Inorganic Crystals," submitted to IOP Machine Learning: Science and Technology.

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

2. **Evaluate crystals samples from generative models**

    For each model, run the evaluation script:
    ```bash
    uv run python examples/journal/eval.py --model MODEL_NAME
    ```
    where `MODEL_NAME` is one of `adit`, `cdvae`, `chemeleon`, `chemeleon2`, `diffcsp`, `diffcsppp`, `mattergen`, `test`, `chemeleon2_rl_bsun`, `chemeleon2_rl_csun`, and `chemeleon2_rl_csun_u10`.
    Depending on the capabilities of your machine, each command may take hours or even a whole day to complete. 
    Therefore, we provide pre-computed results in the `results.zip` file. 
    To skip the above commands, simply unzip the file.
    The samples generated from these models are available on [Hugging Face](https://huggingface.co/datasets/masahiro-negishi/xtalmet).
    If you would like to sample from the models by yourself, follow the instructions in their original repositories.
    To try out reinforcement learning on Chemeleon2, please see the instructions [here](https://hspark1212.github.io/chemeleon2/index-2/).


3. **Visualize the results**

    Once you have obtained the experimental results, run the visualisation scripts in 'visualize.ipynb' to generate the figures and tables presented in the paper.
    Please note that the preprocessing part may take roughly 30 minutes in total. 
