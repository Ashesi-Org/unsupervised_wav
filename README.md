# unsupervised_wav

Unsupervised_wav is a collection of scripts that automate running the Fairseq wav2vec 2.0 Unsupervised Speech Recognition pipeline as described in the official Fairseq project:

https://github.com/facebookresearch/fairseq/blob/main/examples/wav2vec/unsupervised/README.md

These scripts have been tested to work reliably in a Python virtual environment with PyTorch == 2.3.0


## System Requirements

Before running the project, ensure the following:

* Linux-based system (recommended)
* NVIDIA GPU with CUDA support
* Python virtual environment (venv)

CUDA Version Requirement
You must install a CUDA version that is compatible with your GPU and PyTorch version.
Use the official NVIDIA CUDA archive to find the correct version for your system:
https://developer.nvidia.com/cuda-toolkit-archive

After identifying the correct CUDA version:

1. Open `setup_functions.sh`
2. Locate the `cuda_installation` function
3. Update the CUDA download links to match your system


All commands below should be executed from a terminal.

### Step 1: Make Scripts Executable

chmod +x setup_functions.sh \
        wav2vec_functions.sh \
        eval_functions.sh \
        run_setup.sh \
        run_wav2vec.sh \
        run_eval.sh


### Step 2: Run Environment Setup

This step installs dependencies, configures Fairseq, and prepares the environment.

./run_setup.sh

### Step 3: Configure GAN Training (Optional but Recommended)

Before running run_wav2vec.sh, you may want to adjust training hyperparameters.

Edit the configuration file:
unsupervised_wav/fairseq/examples/wav2vec/unsupervised/config/gan/w2vu.yaml

You can modify parameters such as:

* Batch size
* Learning rate
* Number of updates
* Discriminator configuration

This step is especially useful when working with low-resource datasets.

### Step 4: Run Wav2Vec Unsupervised Training

(A short description:)

./run_wav2vec.sh "/path/to/train_audio_dataset" \
                "/path/to/val_audio_dataset" \
                "/path/to/unlabelled/text_dataset"


For the scripts to run successfully:

* All audio files must be in .wav format
* Audio files should have consistent sampling rates (recommended: 16 kHz)

### Step 5: Train GANS

### Step 6: Run Evaluation

After training completes, run:

./run_eval.sh

## Self-Training Configuration for Small Datasets

During self-training, Fairseq uses Kaldi-based scripts that assume a minimum dataset size.

If your dataset contains fewer than 5,000 audio files, you must modify specific parameters to avoid runtime errors.
File: fairseq/examples/wav2vec/unsupervised/kaldi_self_train/st/train.sh


### Default Parameters (For Large Datasets)

local/train_subset_lgbeam.sh \
  --out_root ${out_dir} --out_name exp --train $train_name --valid $valid_name \
  --mono_size 2000 --tri1_size 5000 --tri2b_size -1 --tri3b_size -1 \
  --stage 1 --max_stage 3 $data_dir $data_dir/lang $data_dir/lang_test


### Recommended Changes for Small Datasets (< 5k Audios)

Set the subset sizes to -1 so that all available audio data is used:

local/train_subset_lgbeam.sh \
  --out_root ${out_dir} --out_name exp --train $train_name --valid $valid_name \
  --mono_size -1 --tri1_size -1 --tri2b_size -1 --tri3b_size -1 \
  --stage 1 --max_stage 3 $data_dir $data_dir/lang $data_dir/lang_test


### Explanation

* mono_size and tri1_size control how much data is used during early GMM-HMM training stages
* Setting them to -1 disables sub-sampling
* This ensures all your available audio data is used, which is critical for low-resource settings


## Summary

1. Install the correct CUDA version for your GPU
2. Make all scripts executable
3. Run `run_setup.sh`
4. Ensure all audio files are `.wav`
5. Optionally adjust GAN training configs
6. Run `run_wav2vec.sh`
7. Modify self-training parameters if dataset < 5k
8. Run `run_eval.sh`

