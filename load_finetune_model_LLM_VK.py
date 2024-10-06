import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class FineTunedModelLoader_LLM_VK:
    """
    A class to load and generate text using a fine-tuned LLM model.
    
    Attributes:
        model_path (str): Path to the fine-tuned model.
        model (AutoModelForCausalLM): The loaded model instance.
        tokenizer (AutoTokenizer): The tokenizer for processing input and output text.
        device (torch.device): Device (CPU/GPU) where the model will be loaded.
    """

    def __init__(self, model_path='./finetuned_open_llama_3b_v2'):
        """
        Initializes FineTunedModelLoader_LLM_VK with the provided model path and sets up the device.

        Args:
            model_path (str): Path to the fine-tuned model. Defaults to './finetuned_open_llama_3b_v2'.
        """
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self):
        """
        Loads the fine-tuned model and tokenizer from the specified path.

        Raises:
            OSError: If the model or tokenizer cannot be loaded from the specified path.
        """
        print(f"Loading tokenizer from: {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        print(f"Loading model from: {self.model_path}")
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
            self.model.to(self.device)
            print("Model loaded successfully!")
        except OSError as e:
            print(f"Failed to load model from {self.model_path}: {e}")
            raise

    def generate_text(self, instruction, max_length=100, num_return_sequences=1):
        """
        Generates text based on the given instruction using the fine-tuned model.

        Args:
            instruction (str): Instruction to guide text generation.
            max_length (int): Maximum number of tokens to generate. Default is 100.
            num_return_sequences (int): Number of sequences to generate. Default is 1.

        Returns:
            str: The generated text based on the instruction.

        Raises:
            ValueError: If the model or tokenizer is not loaded before calling this method.
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer are not loaded. Call load_model() first.")

        print(f"Generating text for instruction: {instruction}")

        # Prepare input prompt
        inputs = self.tokenizer(instruction, return_tensors="pt").to(self.device)

        # Generate text using the fine-tuned model
        output_ids = self.model.generate(
            inputs["input_ids"],
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            do_sample=True,
            top_k=50,
            top_p=0.95
        )

        # Decode the generated text from token IDs back to text
        generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f"Generated text: {generated_text}")

        return generated_text
