import torch
from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizerFast
from transformers import AutoModelForCausalLM
import numpy as np
from transformers import Trainer, TrainingArguments
from tdc import Oracle
from typing import List, Any
from tqdm.auto import tqdm
from rdkit import Chem
from rdkit.Chem import Descriptors
import csv
from datetime import datetime
import os
import evaluate
import pandas as pd
from fcd_torch import FCD

def generate_molecules(model, tokenizer, start_tokens, num_samples=1000, max_length=100, batch_size=10, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Generate molecules and log the metrics to a CSV file.

    Args:
        fine_tuned_model: The fine-tuned language model.
    """

    model.to(device)
    input_ids = tokenizer.encode(start_tokens, return_tensors='pt').to(device)
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long).to(device)
    generated_molecules = []

    num_batches = num_samples // batch_size
    for _ in tqdm(range(num_batches), desc="Generating molecules"):
        output = model.generate(
            input_ids.to(device),
            # attention_mask=attention_mask,
            max_length=max_length,
            num_return_sequences=batch_size,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=input_ids[0][0].item(),
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.95
        ).to(device)
        for i in range(batch_size):
            molecule = tokenizer.decode(output[i], skip_special_tokens=True)
            generated_molecules.append(molecule)
    
    return generated_molecules


def generateNlog(fine_tuned_model):
    """
    Generate molecules and log the metrics to a CSV file.

    Args:
        fine_tuned_model: The fine-tuned language model.
    """

    # Ensure model is on the correct device
    fine_tuned_model.to('cuda')
    
    start_tokens = 'CCO'
    # fine_tuned_model = AutoModelForCausalLM.from_pretrained("rest_output_qed_max")
    tokenizer = PreTrainedTokenizerFast.from_pretrained("MolGen/Llama-small-PubChem", use_auth_token='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')
    molecules = generate_molecules(fine_tuned_model, tokenizer, start_tokens, num_samples=100, batch_size=10)

    metrics = {
        'Novelty': 0.0,
        'Valid': 0.0,
        'Unique': 0.0,
        'IntDiv': 0.0,
        'FCD': 0.0,
        'QED': 0.0,
        'LogP': 0.0,
        'Penalized LogP': 0.0,
        'SA': 0.0,
        'SCScore': 0.0,
        'SYBA': 0.0,
        'FCD2': 0.0,
    }

    if molecules:
        try:
            gensmi = [Chem.MolToSmiles(mol) for mol in molecules]
            metrics = molevalmetric.compute(gensmi=gensmi, trainsmi=gensmi)
            fcd = FCD(device='cuda', n_jobs= n_jobs)
            fcd_val = fcd(gen, train)
        except Exception as e:
            print(f"An error occurred during metric computation: {e}")

    # Write metrics to CSV
    with open('dummy.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            'timestamp', 'Novelty', 'Valid', 'Unique', 'IntDiv', 'FCD', 
            'QED', 'LogP', 'Penalized LogP', 'SA', 'SCScore', 'SYBA', 'FCD2'
        ])
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        
        writer.writerow([
            timestamp, metrics['Novelty'], metrics['Valid'], metrics['Unique'], 
            metrics['IntDiv'], metrics['FCD'], metrics['QED'], metrics['LogP'], 
            metrics['Penalized LogP'], metrics['SA'], metrics['SCScore'], metrics['SYBA'],
            fcd_val
        ])


# Set Hugging Face token
os.environ["HUGGINGFACE_TOKEN"] = "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"

# Set CUDA_VISIBLE_DEVICES to use a specific GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Change this to an available GPU
torch.cuda.empty_cache()

# Function to remove invalid molecules
def remove_invalid(gen: List[str], n_jobs: int = 1) -> List[str]:
    """
    Remove invalid molecules from the generated list.

    Args:
        gen: List of generated molecules as SMILES strings.
        n_jobs: Number of parallel jobs for processing.

    Returns:
        List[str]: List of valid molecules as SMILES strings.
    """

    valid_molecules = []
    for smi in gen:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            valid_molecules.append(smi)
    return valid_molecules

# Function to calculate the fraction of unique molecules
def fraction_unique(gen: List[str], num_sample: int = 10000, n_jobs: int = 1) -> float:
    """
    Calculate the fraction of unique molecules in the generated list.

    Args:
        gen: List of generated molecules as SMILES strings.
        num_sample: Number of samples to consider for uniqueness calculation.
        n_jobs: Number of parallel jobs for processing.

    Returns:
        float: Fraction of unique molecules.
    """
    sample = gen[:num_sample]
    unique_molecules = set(sample)
    return len(unique_molecules) / len(sample)

class ReSTTrainer:
    def __init__(
        self,
        model: Any,
        tokenizer: PreTrainedTokenizerFast,
        device: torch.device = torch.device("cuda"),
        num_workers: int = 8,
    ) -> None:
        """Initialize a `MNISTLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """

        self.num_workers = num_workers
        self.device = device
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.oracle = Oracle(name='QED')
        self.prev_train_outputs = None  # Initialize here
        self.mean_reward = 0.0  # Initialize here

        # CSV logging setup
        self.log_file = 'training_dummy.csv'
        with open(self.log_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                'timestamp', 'mean_reward', 'train_runtime', 
                'train_samples_per_second', 'train_steps_per_second', 
                'train_loss', 'total_flos', 'epoch'
            ])
        with open('dummy.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                'timestamp', 'Novelty', 'Valid', 'Unique', 'IntDiv', 'FCD', 
                'QED', 'LogP', 'Penalized LogP', 'SA', 'SCScore', 'SYBA'
            ])

    def reward_model(self, molecules: List[str]) -> np.ndarray:
        """
        Calculate rewards for the generated molecules.

        Args:
            molecules: List of generated molecules as SMILES strings.

        Returns:
            np.ndarray: Array of rewards for each molecule.
        """

        valid_molecules = remove_invalid(molecules)
        invalid_molecules_count = len(molecules) - len(valid_molecules)
        valid_scores = np.asarray(self.oracle(valid_molecules))

        # Penalize for invalid molecules
        penalty = 0.0  # Define the penalty value for invalid molecules
        invalid_penalty = penalty * invalid_molecules_count

        rewards = np.zeros(len(molecules))
        valid_idx = 0
        for i, mol in enumerate(molecules):
            if mol in valid_molecules:
                rewards[i] = valid_scores[valid_idx] * 5.0
                valid_idx += 1
            else:
                rewards[i] = invalid_penalty

        return -rewards

    def log_training_metrics(self):
        """
        Log training metrics to a CSV file.
        """

        metrics = self.prev_train_outputs
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(self.log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                timestamp, self.mean_reward, metrics['train_runtime'], 
                metrics['train_samples_per_second'], metrics['train_steps_per_second'], 
                metrics['train_loss'], metrics['total_flos'], metrics['epoch']
            ])

    def grow_step(
        self,
        model: Any,
        n_samples: int = 1024,
        num_return_sequences: int = 1024,
        no_repeat_ngram_size: int = 2,
        prompt: str = "CCO",
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
    ) -> List[str]:
        """
        Generate sequences using the model and tokenize them. 
        Args:
            model: The language model to use for generation.
            n_samples: Number of samples to generate.
            num_return_sequences: Number of sequences to return per generation call.
            no_repeat_ngram_size: Size of n-grams to avoid repeating.
            prompt: Initial prompt for generation.
            temperature: Sampling temperature for generation.
            top_k: Top-k sampling parameter.
            top_p: Top-p sampling parameter.

        Returns:
            List[str]: List of valid generated sequences.        
        """

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)  # type: ignore
        # skip eos token
        input_ids = input_ids[:, :-1]
        generated_sequences = []
        for _ in tqdm(range(n_samples // num_return_sequences)):
            
            output = model.generate(
                input_ids,
                max_length=64,
                num_return_sequences=num_return_sequences,
                no_repeat_ngram_size=no_repeat_ngram_size,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                return_dict_in_generate=True,
            )
            output = self.tokenizer.batch_decode(output.sequences, skip_special_tokens=True)  # type: ignore
            generated_sequences += [s.replace(" ", "")  for s in output]
        
        valid_molecules = remove_invalid(gen=generated_sequences, n_jobs=self.num_workers)

        metrics = {
            'Novelty': 0.0,
            'Valid': 0.0,
            'Unique': 0.0,
            'IntDiv': 0.0,
            'FCD': 0.0,
            'QED': 0.0,
            'LogP': 0.0,
            'Penalized LogP': 0.0,
            'SA': 0.0,
            'SCScore': 0.0,
            'SYBA': 0.0,
            'FCD2':0,
        }

        if valid_molecules:
            print(f"Valid molecules: {len(valid_molecules)}")
            print(f"Fraction unique: {fraction_unique(valid_molecules, 10000, n_jobs=self.num_workers)}")
            
        try:
            gensmi = generate_molecules(model = model, tokenizer = self.tokenizer, start_tokens = 'CCO', num_samples=5000, max_length=100, batch_size=200)
            print('len(gensmi): ', len(gensmi))
            # import sys
            # sys.exit()
            valid_gensmi = remove_invalid(gen=gensmi, n_jobs=self.num_workers)
            print('len(valid_gensmi): ', len(valid_gensmi))
            df = pd.read_csv('/mnt/media/aman/aditya/models/RainDiffusion/train/molgpt/datasets/moses2.csv')
            valid_trainsmi = df['SMILES'].tolist()
            valid_trainsmi = valid_trainsmi[0:len(valid_gensmi)]
            molevalmetric = evaluate.load('saicharan2804/molgenevalmetric')
            metrics = molevalmetric.compute(gensmi=valid_gensmi, trainsmi=valid_trainsmi)
            print(metrics)
        except Exception as e:
            print(f"An error occurred during metric computation: {e}")
            import sys
            sys.exit()

        # Write metrics to CSV
        with open('generated_metrics_qed_max_5k.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            writer.writerow([
                timestamp, metrics['Novelty'], metrics['Valid'], metrics['Unique'], 
                metrics['IntDiv'], metrics['FCD'], metrics['QED'], metrics['LogP'], 
                metrics['Penalized LogP'], metrics['SA'], metrics['SCScore'], metrics['SYBA']
            ])

        return valid_molecules

    def improve_step(self, model: Any, generated_sequences: List[str], step: int):
        """
        Fine-tune the model with the generated sequences.

        Args:
            model: The language model to fine-tune.
            generated_sequences: List of generated sequences for fine-tuning.
            step: Current training step.

        Returns:
            Any: Fine-tuned model.
        """

        if not generated_sequences:
            print("No generated sequences to process. Skipping improve step.")
            return model

        tokenized_dataset = self.tokenizer(generated_sequences)["input_ids"]
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False
        )

        args = TrainingArguments(
            output_dir="rest_output",
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            evaluation_strategy="steps",
            logging_steps=5_000,
            gradient_accumulation_steps=8,
            num_train_epochs=15,
            weight_decay=0.1,
            warmup_steps=1_000,
            lr_scheduler_type="cosine",
            learning_rate=5e-4,
            bf16=True,
            push_to_hub=False,
            do_eval=False,
        )

        trainer = Trainer(
            model=model,
            tokenizer=self.tokenizer,
            args=args,
            data_collator=data_collator,
            train_dataset=tokenized_dataset,  # type: ignore
        )
        train_output = trainer.train()
        self.prev_train_outputs = train_output.metrics
        return trainer.model

    def train(self, grow_step: int, improve_step: int):
        """
        Train the model using grow and improve steps.

        Args:
            grow_step: Number of grow steps to perform.
            improve_step: Number of improve steps to perform.

        Returns:
            Any: Trained model.
        """
        
        step = 0
        for i in range(grow_step):
            if i == 0:
                # Use the initial model for the first step
                molecules = self.grow_step(self.model)
            else:
                # Use the trained model for subsequent steps
                molecules = self.grow_step(trained_model)  # type: ignore
            rewards = self.reward_model(molecules)
            self.mean_reward = rewards.mean()

            

            percentile = 50
            for _ in range(improve_step):
                threshold = np.percentile(rewards, percentile)
                selected_molecules = [
                    mol for mol, reward in zip(molecules, rewards) if reward > threshold
                ]

                if i == 0:
                    trained_model = self.improve_step(self.model, selected_molecules, step)
                else:
                    trained_model = self.improve_step(trained_model, selected_molecules, step)
                percentile += 5
                step += 1
                self.log_training_metrics()
            # generateNlog(trained_model)
        return trained_model

if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained("MolGen/Llama-small-PubChem", token='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')
    tokenizer = PreTrainedTokenizerFast.from_pretrained("MolGen/Llama-small-PubChem", token='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')

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # Update the model to accommodate the new token
    model.resize_token_embeddings(len(tokenizer))

    trainer = ReSTTrainer(model, tokenizer)
    trained_model = trainer.train(grow_step=10, improve_step=5)
    trained_model.save_pretrained("rest_output_qed_max")
