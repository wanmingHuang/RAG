# RAG Framework and Fine-tuning Generator

This repository contains scripts for constructing, running, and evaluating Retrieval-Augmented Generation (RAG) frameworks, as well as scripts for fine-tuning the generator framework. The project is designed to facilitate research and development in the area of question answering systems and other NLP applications that can benefit from retrieval-augmented generation models.

## Overview

The project includes two main scripts:
- `run_retrieval_logics.py`: Constructs, runs, and evaluates RAG frameworks. The configurations for each RAG framework are defined in YAML files located in the `cfgs/` directory.
- `run_fine_tune_model.py`: Fine-tunes the generator framework. This script supports both training and inference modes.

## Getting Started

Follow these instructions to set up the project environment and run the scripts for your purposes.

### Prerequisites

Before running the scripts, ensure you have Python and the necessary packages installed.

```bash
pip install -r requirements.txt
```

### RAG framework

To construct an RAG framework, specify its settings in a yaml file, an example is [cfg.yaml](cfgs/cfg.yaml).

e.g., execute the following to create 2 frameworks from cfg.yaml and cfg_v2.yaml, execute both and report the performance using ROUGE.

```bash
python run_retrieval_logics.py --config_files cfgs/cfg.yaml cfgs/cfg_v2.yaml 
```

### Fine-tuning Generator

The file is structured around two main components:

- `Trainer`: A class responsible for setting up and executing the fine-tuning process for a causal language model.
- `Inferencer`: A class designed for loading a fine-tuned model and performing inference to generate text based on input prompts.

To run this Generator, first specify its settings in a yaml file, an example is [fine_tune_cfg.yaml](fine_tune_cfg.yaml).

To start the training process, run the script with the --mode argument set to train:

```bash
python run_fine_tune_model.py --mode train --config_file cfgs/fine_tune_cfg.yaml 
```

For performing inference with a trained model, run the script with the --mode argument set to inference and optionally provide a question with --eval_q:

```bash
python run_fine_tune_model.py --mode inference --config_file cfgs/fine_tune_cfg.yaml --eval_q "How do I claim expense?"
```