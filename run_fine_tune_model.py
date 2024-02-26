import argparse
from datasets import load_dataset
from peft import prepare_model_for_kbit_training, PeftModel, LoraConfig, get_peft_model
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
from dotenv import load_dotenv
from typing import List, Union

from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from rag_logger import RAGLogger

from utils import load_config

logger = RAGLogger(__name__).get_logger()

load_dotenv()


class Trainer:
    """Handles the setup and execution of model training.

    This class manages the training process of a causal language model by loading configuration,
    preparing data, setting up tokenizer and model, and finally executing the training routine.

    Attributes:
        config_path (str): Path to the configuration file.
        cfg (dict): Configuration loaded from the config file.
        base_model_id (str): Identifier for the base pre-trained model.
        train_data_file (str): Path to the training data file.
        eval_data_file (str): Path to the evaluation data file.
        tokenizer (AutoTokenizer): Tokenizer for the model.
        model (AutoModelForCausalLM): The language model for training.
        dataset (Dataset): The dataset for training and evaluation.
        trainer (Trainer): The Hugging Face Trainer instance for model training.
    """
    def __init__(self, config_path):
        self.config_path = config_path
        self.cfg = load_config(config_path)

        self.base_model_id = self.cfg['base_model']
        self.train_data_file = self.cfg['train']['data']
        self.eval_data_file = self.cfg['eval']['data']
        self.tokenizer = None
        self.model = None
        self.dataset = None
        self.trainer = None

    def load_and_prepare_data(self):
        """Loads and prepares training and evaluation data from CSV files.

        This method loads the training and evaluation data specified in the configuration, applies tokenization
        to the data using the `generate_and_tokenize_prompt` method, and stores the processed datasets for training and evaluation.
        """
        data_files = {'train': self.train_data_file, 'eval': self.eval_data_file}
        self.dataset = load_dataset('csv', data_files=data_files).map(self.generate_and_tokenize_prompt)

    def setup_tokenizer(self):
        """Initializes the tokenizer for the model.

        Sets up the tokenizer with appropriate configuration for padding and token addition. The tokenizer's pad token is set to the EOS token.
        """
        logger.info('set up tokenizer')
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_id,
            padding_side="left",
            add_eos_token=True,
            add_bos_token=True,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def setup_model(self):
        """Prepares the model for training with specific configurations.

        This method initializes the model for causal language modeling from a pre-trained base, enables gradient checkpointing,
        and prepares it for k-bit training using PEFT configurations.
        """
        logger.info('set up model')
        self.model = AutoModelForCausalLM.from_pretrained(self.base_model_id, device_map="cpu")
        self.model.gradient_checkpointing_enable()
        self.model = prepare_model_for_kbit_training(self.model)

        config = LoraConfig(
            target_modules=["lm_head"],
            bias="none",
            task_type="CAUSAL_LM",
        )

        self.model = get_peft_model(self.model, config)

    def generate_and_tokenize_prompt(self, example: dict):
        """Generates and tokenizes prompts from dataset examples.

        Args:
            example (dict): A single example from the dataset.

        Returns:
            dict: Tokenized input suitable for model training or evaluation.
        """

        formatted_text = f"Question: {example['Question']}\n Answer: {example['Answer']}"
        return self.tokenizer(formatted_text)

    def setup_trainer(self):
        """Configures the Trainer for model training.

        Initializes the Trainer with model, dataset, training arguments, and data collator.
        This setup includes configuration for output directory, learning rate, evaluation frequency, and saving checkpoints.
        """
        logger.info("set up trainer")

        self.trainer = transformers.Trainer(
            model=self.model,
            train_dataset=self.dataset['train'],
            eval_dataset=self.dataset['eval'],
            args=transformers.TrainingArguments(
                output_dir=self.cfg['train']['save_dir'],
                max_steps=self.cfg['train']['max_steps'],
                learning_rate=self.cfg['train']['lr'],
                save_steps=self.cfg['train']['max_steps'],
                eval_steps=self.cfg['train']['eval_steps'],
                do_eval=True,
            ),
                data_collator=transformers.DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
            )
        
    def train_model(self):
        """Executes the model training process.

        Orchestrates the training workflow by setting up the tokenizer, data, model, and trainer before starting the training process.
        """
        self.setup_tokenizer()
        self.load_and_prepare_data()
        self.setup_model()
        self.setup_trainer()
        self.trainer.train()

class Inferencer:
    """Manages model inference, loading models, and parsing output.

    This class is designed to perform inference with a pre-trained model. It loads the model and tokenizer configurations,
    prepares the model for inference, and processes the output using a structured output parser.

    Attributes:
        config_path (str): Path to the configuration file.
        cfg (dict): Configuration loaded from the config file.
        tokenizer (AutoTokenizer): Tokenizer for the model.
        model (PeftModel): The PEFT-enhanced model for inference.
        output_parser (StructuredOutputParser): Parser for structured output from model predictions.
    """
    def __init__(self, config_path):
        self.config_path = config_path
        self.cfg = load_config(config_path)

        self.setup_tokenizer()
        self.setup_model()
        self.setup_output_parser()

    def setup_model(self):
        """load pretrained model
        """
        base_model = AutoModelForCausalLM.from_pretrained(self.cfg['base_model'], device_map="cpu")
        self.model = PeftModel.from_pretrained(base_model, self.cfg['inf']['ckpt'])

    def setup_tokenizer(self):
        """set up tokenizer from pre-trained model
        """
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg['base_model'], add_bos_token=True, trust_remote_code=True)

    def setup_output_parser(self):
        """Initializes the output parser for structured model predictions.

        Configures the output parser with response schemas to structure the model's output for consumption.
        """
        response_schemas = [
            ResponseSchema(name="answer", description="answer")
        ]
        self.output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

    def __call__(self, eval_prompt: Union[str, List[str]]) -> str:
        """Performs inference on the provided evaluation prompt.

        Args:
            eval_prompt (str): The input text prompt for evaluation.

        Returns:
            str: The decoded model output, post-processed to remove the input prompt part.
        """
        if not isinstance(eval_prompt, str):
            eval_prompt = eval_prompt.to_string()
        model_input = self.tokenizer(eval_prompt, return_tensors="pt")

        self.model.eval()
        with torch.no_grad():
            return self.tokenizer.decode(
                self.model.generate(**model_input, max_new_tokens=self.cfg['inf']['max_new_token'], pad_token_id=self.tokenizer.eos_token_id)[0],
                skip_special_tokens=True
            )[len(eval_prompt):]

def main():
    """The mainer function for fine-tuning
    """
    # Create the argument parser
    parser = argparse.ArgumentParser(description='Process command line arguments.')

    # Add arguments
    parser.add_argument('--mode', type=str, choices=['train', 'inference'], help='Mode: train or inference')
    parser.add_argument('--config_file', type=str, help='config file for fine tuning')
    parser.add_argument('--eval_q', type=str, default="", required=False, help='Ask a question')

    # Parse arguments
    args = parser.parse_args()

    if args.mode == 'train':
        logger.info("Training mode selected.")
        trainer = Trainer(args.config_file)
        trainer.train_model()
    elif args.mode == 'inference':
        logger.info("Inference mode selected.")
        inferencer = Inferencer(args.config_file)
        ans = inferencer(args.eval_q)
        logger.info(ans)
    else:
        raise Exception("Requested mode unknown")
    

if __name__ == "__main__":
    main()