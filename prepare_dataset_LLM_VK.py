from datasets import load_dataset

class DatasetPreparator_LLM_VK:
    """
    A class to load and process the Alpaca dataset for use with LLM models.
    
    Attributes:
        dataset (DatasetDict): The loaded dataset.
    """

    def __init__(self):
        """
        Initializes the DatasetPreparator_LLM_VK with no dataset loaded.
        """
        self.dataset = None

    def load_alpaca_dataset(self):
        """
        Loads the Alpaca dataset from Hugging Face Datasets library.

        Returns:
            DatasetDict: The loaded dataset.
        """
        print("Loading the Alpaca dataset...")
        self.dataset = load_dataset("mhenrichsen/alpaca_2k_test")
        print("Dataset loaded successfully!")
        return self.dataset

    def print_sample(self, split="train", num_samples=5):
        """
        Prints a specified number of samples from the specified dataset split.

        Args:
            split (str): The dataset split to print samples from (default is 'train').
            num_samples (int): The number of samples to print (default is 5).
        """
        if self.dataset is None:
            print("Dataset not loaded. Please call load_alpaca_dataset() first.")
            return

        print(f"Printing {num_samples} samples from the {split} split:")
        for i, example in enumerate(self.dataset[split].select(range(num_samples))):
            print(f"\nSample {i + 1}:")
            print(example)  # Print the raw example to see its structure

    def get_dataset_info(self):
        """
        Prints information about the loaded dataset, including splits and features.
        """
        if self.dataset is None:
            print("Dataset not loaded. Please call load_alpaca_dataset() first.")
            return

        print("Dataset Information:")
        print(f"Splits: {self.dataset.keys()}")
        for split in self.dataset.keys():
            print(f"{split} split size: {len(self.dataset[split])}")
        print(f"Features: {self.dataset['train'].features}")

if __name__ == "__main__":
    preparator = DatasetPreparator_LLM_VK()
    dataset = preparator.load_alpaca_dataset()
    preparator.get_dataset_info()
    preparator.print_sample()
