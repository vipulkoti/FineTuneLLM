from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class ModelLoader_LLM_VK:
    """
    A class to load and generate text using a pretrained LLM model.
    
    Attributes:
        model_name (str): Name of the pretrained model to load.
        model (AutoModelForCausalLM): The loaded model instance.
        tokenizer (AutoTokenizer): The tokenizer for processing input and output text.
    """

    def __init__(self, model_name='openlm-research/open_llama_3b_v2'):
        """
        Initializes the ModelLoader_LLM_VK with the specified model name.
        
        Args:
            model_name (str): The name of the model to load. Defaults to 'openlm-research/open_llama_3b_v2'.
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """
        Loads the model and tokenizer from the specified model name.
        
        Returns:
            tuple: The loaded model and tokenizer.
        """
        print(f"Loading model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        print("Model loaded successfully!")
        return self.model, self.tokenizer

    def generate_text(self, instruction, input_text="", max_new_tokens=100):
        """
        Generates text from the model based on the provided instruction and optional input text.
        
        Args:
            instruction (str): The instruction guiding text generation.
            input_text (str): Additional context input (optional).
            max_new_tokens (int): Maximum number of new tokens to generate. Defaults to 100.
        
        Returns:
            str: The generated text.
        
        Raises:
            ValueError: If the model and tokenizer are not loaded before generating text.
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer are not loaded. Call load_model() first.")
        
        inputs = self.tokenizer(instruction, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.7,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return generated_text

if __name__ == "__main__":
    loader = ModelLoader_LLM_VK()
    model, tokenizer = loader.load_model()
    
    # Test the model with a sample prompt
    instruction = "Explain the concept of machine learning in simple terms"
    generated_text = loader.generate_text(instruction)
    print(f"Instruction: {instruction}")
    print(f"Generated text: {generated_text}")
