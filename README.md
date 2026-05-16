# Running the Disfluency Detection Models (`jk/` Directory)

This project uses the `ling230` Python environment. An `environment.yml` file has been provided in the root directory to easily install all required dependencies.

## 1. Setup the Environment

You can create and activate the conda environment using the provided `environment.yml` file:

```bash
conda env create -f environment.yml
conda activate ling230
```

## 2. Running the Notebooks

The `jk/` directory contains Jupyter Notebooks for training and evaluating different disfluency detection models. To run them, launch Jupyter Notebook or Jupyter Lab from the root directory:

```bash
jupyter notebook
```

Navigate to the `jk/` directory in the Jupyter interface and open any of the following notebooks to run them sequentially:

- `cnn-transformer.ipynb`: The main CNN-Transformer model pipeline.
- `block.ipynb`: Model focused on detecting "block" disfluencies.
- `Prolongation.ipynb`: Model focused on detecting "prolongation" disfluencies.
- `SoundsRep.ipynb`: Model focused on detecting "sound repetition" disfluencies.

> **Note on Data**: The notebooks rely on pre-computed Whisper embeddings stored as `.npy` files and path mappings in `.csv` files within the `jk/` directory. Ensure these large data files are present before running the cells.
if you ran the numpy array once its saved you can skip that cell if no changes were made to the parameter before. This allows modifications to be made to the model after the embeddings are computed.

## 3. Running Multi-Task Approaches (`ksh/` Directory)

The `ksh/` directory contains experiments for multi-task learning, specifically within the `ksh/multitask/` folder. 

To run these models, navigate to the `ksh/multitask/` directory in Jupyter and open any of the following notebooks:
- `full_model.ipynb`: The complete multi-task model pipeline.
- `data-sampling.ipynb`: Data sampling and processing.
- Various ablation studies (e.g., `no-block.ipynb`, `no-pro.ipynb`, `no-sr.ipynb`) representing different multi-task model configurations.

**Testing**
To run the evaluation on new unseen data follow the steps mentioned in the `testing/read.txt` file.
