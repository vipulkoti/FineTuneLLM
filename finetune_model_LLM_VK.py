import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from datasets import Dataset
from trl import SFTTrainer

class ModelFinetuner_LLM_VK:
    """
    A class for finetuning a pretrained LLM model using PEFT and LoRA techniques.
    
    Attributes:
        model_name (str): The name of the pretrained model to load and finetune.
        dataset (Dataset): The dataset to use for finetuning.
        model (AutoModelForCausalLM): The model loaded for finetuning.
        tokenizer (AutoTokenizer): The tokenizer used for processing text inputs.
    """

    def __init__(self, model_name='openlm-research/open_llama_3b_v2', dataset=None):
        """
        Initializes ModelFinetuner_LLM_VK with a model name and dataset.

        Args:
            model_name (str): The name of the pretrained model to load. Defaults to 'openlm-research/open_llama_3b_v2'.
            dataset (Dataset): The dataset to use for finetuning. Defaults to None.
        """
        self.model_name = model_name
        self.dataset = dataset
        self.model = None
        self.tokenizer = None

    def prepare_model(self):
        """
        Prepares the model and tokenizer for finetuning. The model is loaded with quantization, 
        and configured for k-bit training with LoRA applied.
        """
        print(f"Preparing model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Set the padding token to the eos token if it's not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print("Padding token set to EOS token.")
        
        # Load the model with 4-bit quantization
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            device_map="auto"
        )
        self.model = prepare_model_for_kbit_training(self.model)

        # Apply LoRA configuration
        config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.model = get_peft_model(self.model, config)
        print("Model prepared successfully!")

    def prepare_dataset(self):
        """
        Tokenizes and prepares the dataset for training.

        Returns:
            Dataset: The tokenized dataset ready for training.

        Raises:
            ValueError: If the dataset is not provided.
        """
        if self.dataset is None:
            raise ValueError("Dataset not provided. Please set the dataset before calling this method.")

        def tokenize_function(examples):
            prompts = [
                f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
                for instruction, input_text, output in zip(examples['instruction'], examples['input'], examples['output'])
            ]
            tokenized = self.tokenizer(prompts, truncation=True, padding="max_length", max_length=512)
            return tokenized

        tokenized_dataset = self.dataset.map(tokenize_function, batched=True, remove_columns=self.dataset.column_names)
        return tokenized_dataset

    def train(self, output_dir="./finetuned_open_llama_3b_v2", num_train_epochs=3, per_device_train_batch_size=4):
        """
        Trains the model on the provided dataset.

        Args:
            output_dir (str): Directory to save the finetuned model. Defaults to './finetuned_open_llama_3b_v2'.
            num_train_epochs (int): Number of training epochs. Defaults to 3.
            per_device_train_batch_size (int): Batch size per device. Defaults to 4.

        Raises:
            ValueError: If the model and tokenizer are not prepared before training.
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer are not prepared. Call prepare_model() first.")

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=4,
            optim="paged_adamw_32bit",
            save_steps=100,
            logging_steps=100,
            learning_rate=2e-4,
            weight_decay=0.001,
            fp16=True,
            bf16=False,
            max_grad_norm=0.3,
            max_steps=-1,
            warmup_ratio=0.03,
            group_by_length=True,
            lr_scheduler_type="constant",
        )

        # Configure PEFT parameters for training
        peft_params = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=64,
            bias="none",
            task_type="CAUSAL_LM",
        )

        trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.prepare_dataset(),
            tokenizer=self.tokenizer,
            peft_config=peft_params
        )

        print("Starting training...")
        trainer.train()
        print("Training completed!")

        # Save the final model and tokenizer
        trainer.model.save_pretrained(output_dir)
        trainer.tokenizer.save_pretrained(output_dir)
        print(f"Model saved to {output_dir}")

if __name__ == "__main__":
    # Placeholder dataset for demonstration
    dummy_dataset = Dataset.from_dict({
        "instruction": ["Write a poem", "Explain quantum physics"],
        "input": ["About love", "In simple terms"],
        "output": ["Roses are red...", "Quantum physics is..."]
    })
    
    finetuner = ModelFinetuner_LLM_VK(dataset=dummy_dataset)
    finetuner.prepare_model()
    finetuner.train()
