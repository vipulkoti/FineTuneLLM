# Fine-tuning OpenLLaMA 3B V2

This repository contains code for fine-tuning the **OpenLLaMA 3B V2** model using the **Hugging Face dataset** `mhenrichsen/alpaca_2k_test`. The goal of this project is to enhance the model's ability to generate high-quality responses based on specific instructions.

## Overview

- **Model**: [openlm-research/open_llama_3b_v2](https://huggingface.co/openlm-research/open_llama_3b_v2)
- **Dataset**: [mhenrichsen/alpaca_2k_test](https://huggingface.co/datasets/mhenrichsen/alpaca_2k_test)
- **Framework**: Hugging Face Transformers, PEFT (Parameter-Efficient Fine-Tuning)

### Key Features

- Preparation of the model and tokenizer for fine-tuning.
- Loading and preprocessing of the Alpaca dataset.
- Configuration of the training process using LoRA (Low-Rank Adaptation).
- Efficient training setup with gradient accumulation and mixed precision.

## Requirements

To run the code in this repository, you need to have the following packages installed:

```bash
pip install torch transformers datasets peft trl
```
## File Structure
```bash
FineTuneLLM /
├── test_load_model_LLM_VK.py          # Contains unit tests for loading models
├── prepare_dataset_LLM_VK.py           # Contains functions for preparing the dataset
├── load_model_LLM_VK.py                # Contains the ModelLoader_LLM_VK class for loading models
├── load_finetune_model_LLM_VK.py       # Contains the code for loading the fine-tuned model
├── finetune_model_LLM_VK.py            # Contains the ModelFinetuner_LLM_VK class for fine-tuning the model
├── FineTune_Llama_LLM_VK.ipynb          # Jupyter Notebook for interactive model fine-tuning
└── README.md                            # This README file

```

## Usage

1. **Clone the Repository**:  
   To clone the repository, run the following command in your terminal:
   ```bash
   git clone https://github.com/your_username/your_repo.git
   cd your_repo
   ```

2. **Prepare the Dataset**: Create a dataset using the Alpaca dataset.
   ```python
   from datasets import Dataset

   # Example of creating a dataset
   dummy_dataset = Dataset.from_dict({
       "instruction": ["Write a poem", "Explain quantum physics"],
       "input": ["About love", "In simple terms"],
       "output": ["Roses are red...", "Quantum physics is..."]
   })

  3. **Fine-Tune the Model**:  
   To fine-tune the model with the prepared dataset, use the following code:
   ```python
   from finetune_model_LLM_VK import ModelFinetuner_LLM_VK

   finetuner = ModelFinetuner_LLM_VK(dataset=dummy_dataset)
   finetuner.prepare_model()
   finetuner.train()
```
## Training Configuration

The training parameters can be adjusted in the `train` method of the `ModelFinetuner_LLM_VK` class. Key parameters include:

- `output_dir`: Directory where the fine-tuned model will be saved.
- `num_train_epochs`: Number of epochs for training.
- `per_device_train_batch_size`: Batch size per device during training.

## License

This project is for education purpose only.

## Acknowledgments

- [Hugging Face](https://huggingface.co/) for their excellent library and resources.
- [OpenLLaMA](https://huggingface.co/openlm-research/open_llama_3b_v2) for providing the model.

