# Child-ASR-Word-Benchmark

This repository contains a reference implementation for the **Word Track** of the [On Top of Pasketti: Children’s Speech Recognition Challenge](https://www.drivendata.org/competitions/308/childrens-word-asr/) hosted by DrivenData.

## Project Overview
Automatic Speech Recognition (ASR) systems often struggle with child speech due to unique pitch, rhythm, and evolving articulation. This project aims to improve child-focused ASR by fine-tuning a state-of-the-art model to accurately transcribe words from audio clips of children.

The benchmark demonstrates an end-to-end pipeline using **NVIDIA's Parakeet** model via the **NeMo** framework.

## Key Features
* **Data Exploration:** Utilities for loading competition metadata and visualizing audio utterances.
* **Model Architecture:** Adapts a pretrained Parakeet ASR model for child speech characteristics.
* **Training Pipeline:** Configurable training setup using PyTorch Lightning and NeMo.
* **Submission Readiness:** Scripts to package models and inference code for containerized execution.

## Getting Started

### Prerequisites
* Linux machine with GPU access.
* `uv` package manager.

### Installation
1.  **Create the environment:**
    ```bash
    just create-environment
    ```
2.  **Activate the environment:**
    ```bash
    source ./.venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    just requirements
    ```
## Data Structure
The competition utilizes two corpora (DrivenData and TalkBank). The expected raw data structure is:

```text
data/raw
├── drivendata
│   ├── audio/
│   └── train_word_transcripts.jsonl
└── talkbank
    ├── audio/
    └── train_word_transcripts.jsonl


## Pipeline Steps
* **Step 0: Setup** – Environment configuration and library imports.
* **Step 1: Data Exploration** – Analysis of audio distributions and metadata.
* **Step 2: Model Building** – Dataset preparation, trainer definition, and adapter-based fine-tuning.
* **Step 3: Submission** – Local testing of inference code and packaging for the DrivenData runtime.

## Technical Approach:
1. Feature Extraction
The pipeline converts raw audio waveforms into Mel Spectrograms. This transformation reorganizes the signal into a structured time-by-frequency map that highlights speech patterns like vowels and consonants, making it easier for the model to process than raw audio.

2. Adapter-Based Fine-Tuning
Instead of retraining the entire 619M parameter model, we use Linear Adapters inserted into the encoder layers. This keeps the base model frozen and prevents overfitting on the specialized child speech dataset.

Base Model: Frozen (Pretrained nvidia/parakeet-tdt-0.6b-v2).
Trainable Parameters: ~1.6M (0.26% of total).
Optimization: AdamW with Cosine Annealing and 10% warmup.

3. Evaluation
Performance is measured using Word Error Rate (WER). The benchmark implementation achieves:

Validation WER: ~0.15
Full Test Set WER: ~0.2370

Submission Guide
This is a code execution challenge. You must submit a ZIP file containing:

main.py: The entry point script that performs inference in the execution environment.
ASR-Adapter-best.nemo: Your trained model/adapter weights.

Packaging the Submission
Use the provided justfile command to bundle your code and latest weights:

1.  **Pack Submission:**
    ```bash
    just pack-orthographic
    ```

## Technologies Used
* **NVIDIA NeMo:** For ASR model management and fine-tuning.
* **PyTorch Lightning:** For scalable model training.
* **Librosa:** For audio processing and feature extraction.
* **Pandas/NumPy:** For data manipulation.

## License
This project is licensed under the terms of the Apache-2.0 License.

---
*For more details, visit the official [DrivenData Benchmark Blog Post](https://drivendata.co/blog/child-asr-word-benchmark).*
