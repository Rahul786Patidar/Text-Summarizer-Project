import os
from textSummarizer.logging import logger
from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk
from textSummarizer.entity import DataTransformationConfig

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

    def convert_examples_to_features(self, example_batch):
        # Tokenize the input dialogue
        input_encodings = self.tokenizer(example_batch['dialogue'], max_length=1024, truncation=True)

        # Tokenize the target summary
        with self.tokenizer.as_target_tokenizer():
            target_encodings = self.tokenizer(example_batch['summary'], max_length=128, truncation=True)

        # Return the encoded inputs and labels
        return {
            'input_ids': input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
            'labels': target_encodings['input_ids']
        }

    def convert(self):
        try:
            # Load dataset from disk
            dataset_samsum = load_from_disk(self.config.data_path)
            logger.info(f"Dataset loaded from {self.config.data_path}")

            # Apply the tokenization function
            dataset_samsum_pt = dataset_samsum.map(self.convert_examples_to_features, batched=True)
            logger.info("Dataset tokenized successfully")

            # Save the processed dataset to disk
            output_path = os.path.join(self.config.root_dir, "samsum_dataset")
            dataset_samsum_pt.save_to_disk(output_path)
            logger.info(f"Tokenized dataset saved to {output_path}")

        except Exception as e:
            logger.error(f"An error occurred during data transformation: {str(e)}")
            raise e
