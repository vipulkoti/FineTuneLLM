import unittest
from load_model_LLM_VK import ModelLoader_LLM_VK

class TestModelLoader(unittest.TestCase):
    """
    Unit test case for the ModelLoader_LLM_VK class.
    """

    def setUp(self):
        """
        Sets up the ModelLoader_LLM_VK instance for testing.
        """
        self.loader = ModelLoader_LLM_VK()

    def test_model_loading(self):
        """
        Tests whether the model and tokenizer load successfully.
        """
        model, tokenizer = self.loader.load_model()
        self.assertIsNotNone(model, "Model should not be None after loading.")
        self.assertIsNotNone(tokenizer, "Tokenizer should not be None after loading.")

    def test_generate_text(self):
        """
        Tests whether the model can generate text from a given instruction.
        """
        self.loader.load_model()
        instruction = "Explain the concept of machine learning in simple terms"
        generated_text = self.loader.generate_text(instruction)
        
        print(f"Generated text: {generated_text}")
        
        self.assertIsInstance(generated_text, str, "Generated text should be a string.")
        self.assertGreater(len(generated_text), 0, "Generated text should not be empty.")

    def test_generate_text_without_loading(self):
        """
        Tests that generating text without loading the model raises a ValueError.
        """
        loader = ModelLoader_LLM_VK()  # New instance without loading the model
        with self.assertRaises(ValueError, msg="Expected ValueError if model is not loaded before generating text"):
            loader.generate_text("This should raise an error")

if __name__ == '__main__':
    unittest.main()
