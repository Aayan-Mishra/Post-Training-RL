import os
import torch
import numpy as np
import argparse
from torch.optim import Adam
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
import random
import wandb
from torch.utils.data import Dataset, DataLoader
from huggingface_hub import HfApi, Repository, create_repo, upload_file, upload_folder
import shutil
import json
import datetime

# Configuration
@dataclass
class RLConfig:
    model_name: str = "gpt2"                # Pre-trained model to use
    reward_model_name: Optional[str] = None # Separate reward model (if not using a human in the loop)
    learning_rate: float = 1e-5
    batch_size: int = 4
    epochs: int = 3
    max_seq_length: int = 512
    ppo_epochs: int = 4                     # Number of PPO optimization steps per batch
    clip_param: float = 0.2                 # PPO clipping parameter
    value_loss_coef: float = 0.5            # Value function coefficient
    entropy_coef: float = 0.01              # Entropy coefficient for exploration
    max_grad_norm: float = 0.5              # Gradient clipping
    gamma: float = 0.99                     # Discount factor for rewards
    output_dir: str = "rl_finetuned_model"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_wandb: bool = False                 # Whether to log to Weights & Biases
    wandb_project: str = "rl-finetuning"
    human_feedback: bool = False            # Whether to use human feedback instead of a reward model
    hf_repo_name: Optional[str] = None      # HuggingFace repo name for uploading the model
    hf_token: Optional[str] = None          # HuggingFace token for uploading
    upload_to_hub: bool = False             # Whether to upload the model to HuggingFace Hub
    eval_prompts: List[str] = field(default_factory=list)  # Prompts to use for evaluation

# Simple dataset for RL prompts
class RLPromptDataset(Dataset):
    def __init__(self, prompts: List[str], tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.prompts = prompts
        self.max_length = max_length
    
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        tokenized = self.tokenizer(prompt, truncation=True, max_length=self.max_length, 
                                  padding="max_length", return_tensors="pt")
        return {
            "input_ids": tokenized["input_ids"].squeeze(0),
            "attention_mask": tokenized["attention_mask"].squeeze(0),
            "prompt": prompt
        }

# Simple reward model - can be replaced with more sophisticated implementations
class RewardModel:
    def __init__(self, config: RLConfig):
        self.config = config
        if config.reward_model_name:
            # Load pre-trained reward model
            self.model = AutoModelForCausalLM.from_pretrained(config.reward_model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(config.reward_model_name)
            self.model.to(config.device)
            self.model.eval()
        else:
            self.model = None
            self.tokenizer = None
    
    def get_reward(self, prompt: str, response: str) -> float:
        # If using human feedback
        if self.config.human_feedback:
            print("\n" + "="*50)
            print(f"PROMPT: {prompt}")
            print(f"RESPONSE: {response}")
            score = float(input("Rate this response (0-10): "))
            return score
        
        # If using a reward model
        elif self.model:
            # This is a placeholder - actual reward models would have specific implementations
            inputs = self.tokenizer(prompt + response, return_tensors="pt").to(self.config.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            # Use the last token prediction as a simple reward signal
            reward = outputs.logits[0, -1, :].max().item()
            return reward
        
        # Fallback to a simple heuristic reward
        else:
            # Example heuristics:
            # - Length (penalize too short or too long)
            # - Diversity of tokens used
            # - Lack of repetition
            length_score = min(len(response.split()) / 50, 1.0)  # Prefer ~50 word responses
            repetition_penalty = response.count("the the") + response.count("and and")
            return length_score - (0.5 * repetition_penalty)

# PPO Agent
class PPOAgent:
    def __init__(self, config: RLConfig):
        self.config = config
        
        # Load model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(config.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        
        # Initialize reward model
        self.reward_model = RewardModel(config)
        
        # Initialize optimizer
        self.optimizer = Adam(self.model.parameters(), lr=config.learning_rate)
        
        # Move model to device
        self.model.to(config.device)
        
        # For tracking old policy
        self.old_model = None
        
        # For tracking training metrics
        self.training_history = {
            "rewards": [],
            "losses": [],
            "policy_losses": [],
            "entropy_losses": []
        }
    
    def generate_response(self, prompt: str, max_length: int = 50) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.config.device)
        
        # Generate with some randomness for exploration
        outputs = self.model.generate(
            inputs["input_ids"],
            max_length=inputs["input_ids"].shape[1] + max_length,
            do_sample=True,
            temperature=0.7,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        # Decode only the newly generated tokens
        generated = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return generated
    
    def compute_ppo_loss(self, prompt_batch, responses, old_log_probs, rewards):
        total_loss = 0
        value_loss = 0
        policy_loss = 0
        entropy_loss = 0
        
        for i, (prompt, response, old_log_prob, reward) in enumerate(zip(prompt_batch, responses, old_log_probs, rewards)):
            # Tokenize the combined prompt and response
            combined = prompt + response
            tokens = self.tokenizer(combined, return_tensors="pt").to(self.config.device)
            
            # Get current policy logits
            outputs = self.model(**tokens)
            logits = outputs.logits
            
            # Get old policy logits
            with torch.no_grad():
                old_outputs = self.old_model(**tokens)
                old_logits = old_outputs.logits
            
            # Compute log probabilities and ratio for PPO
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            old_log_probs_expanded = torch.nn.functional.log_softmax(old_logits, dim=-1)
            
            # Compute policy ratio (simplified for this example)
            # In practice, you would compute this token by token for the generated sequence
            ratio = torch.exp(log_probs.mean() - old_log_probs_expanded.mean())
            
            # Compute surrogate loss
            surr1 = ratio * reward
            surr2 = torch.clamp(ratio, 1.0 - self.config.clip_param, 1.0 + self.config.clip_param) * reward
            policy_loss_i = -torch.min(surr1, surr2)
            
            # Add entropy bonus for exploration
            entropy = -(log_probs * torch.exp(log_probs)).sum(-1).mean()
            entropy_loss_i = -self.config.entropy_coef * entropy
            
            # Combine losses
            total_loss_i = policy_loss_i + entropy_loss_i
            
            # Accumulate losses
            total_loss += total_loss_i
            policy_loss += policy_loss_i
            entropy_loss += entropy_loss_i
        
        # Average losses
        batch_size = len(prompt_batch)
        total_loss /= batch_size
        policy_loss /= batch_size
        entropy_loss /= batch_size
        
        return total_loss, policy_loss, entropy_loss
    
    def train_step(self, dataloader, epoch):
        self.model.train()
        epoch_stats = {"reward": [], "loss": [], "policy_loss": [], "entropy_loss": []}
        
        for batch in tqdm(dataloader, desc=f"Epoch {epoch}"):
            prompts = batch["prompt"]
            prompt_tensors = batch["input_ids"].to(self.config.device)
            
            # Store current policy
            self.old_model = type(self.model).from_pretrained(self.config.model_name)
            self.old_model.load_state_dict(self.model.state_dict())
            self.old_model.to(self.config.device)
            self.old_model.eval()
            
            # Generate responses
            responses = []
            old_log_probs = []
            rewards = []
            
            for prompt in prompts:
                # Generate response with current policy
                response = self.generate_response(prompt)
                responses.append(response)
                
                # Compute reward
                reward = self.reward_model.get_reward(prompt, response)
                rewards.append(reward)
                
                # Store for logging
                epoch_stats["reward"].append(reward)
                
                # Placeholder for old log probs (would be computed during generation in full implementation)
                old_log_probs.append(0.0)
            
            # Multiple PPO optimization steps on the same batch
            for _ in range(self.config.ppo_epochs):
                # Compute PPO loss
                loss, policy_loss, entropy_loss = self.compute_ppo_loss(prompts, responses, old_log_probs, rewards)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                
                # Log losses
                epoch_stats["loss"].append(loss.item())
                epoch_stats["policy_loss"].append(policy_loss.item())
                epoch_stats["entropy_loss"].append(entropy_loss.item())
        
        # Compute epoch averages
        for key in epoch_stats:
            epoch_stats[key] = sum(epoch_stats[key]) / len(epoch_stats[key]) if epoch_stats[key] else 0
        
        # Add to training history
        self.training_history["rewards"].append(epoch_stats["reward"])
        self.training_history["losses"].append(epoch_stats["loss"])
        self.training_history["policy_losses"].append(epoch_stats["policy_loss"])
        self.training_history["entropy_losses"].append(epoch_stats["entropy_loss"])
        
        return epoch_stats
    
    def evaluate(self, prompts=None):
        """Evaluate the model on a set of prompts"""
        if prompts is None:
            prompts = self.config.eval_prompts if self.config.eval_prompts else [
                "What are the key benefits of reinforcement learning?",
                "Explain how PPO works in simple terms."
            ]
        
        self.model.eval()
        results = []
        
        for prompt in prompts:
            response = self.generate_response(prompt, max_length=100)
            results.append({
                "prompt": prompt,
                "response": response
            })
        
        return results
    
    def train(self, prompts: List[str]):
        # Create dataset and dataloader
        dataset = RLPromptDataset(prompts, self.tokenizer, self.config.max_seq_length)
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        
        # Initialize wandb if requested
        if self.config.log_wandb:
            wandb.init(project=self.config.wandb_project, config=vars(self.config))
        
        # Training loop
        for epoch in range(self.config.epochs):
            stats = self.train_step(dataloader, epoch)
            
            # Log epoch stats
            print(f"Epoch {epoch} stats: {stats}")
            if self.config.log_wandb:
                wandb.log(stats)
            
            # Save checkpoint
            if (epoch + 1) % 5 == 0 or epoch == self.config.epochs - 1:
                if not os.path.exists(self.config.output_dir):
                    os.makedirs(self.config.output_dir)
                checkpoint_path = os.path.join(self.config.output_dir, f"checkpoint-{epoch}")
                self.model.save_pretrained(checkpoint_path)
                self.tokenizer.save_pretrained(checkpoint_path)
        
        # Final save
        if not os.path.exists(self.config.output_dir):
            os.makedirs(self.config.output_dir)
        
        self.model.save_pretrained(self.config.output_dir)
        self.tokenizer.save_pretrained(self.config.output_dir)
        
        # Save training history
        with open(os.path.join(self.config.output_dir, "training_history.json"), "w") as f:
            json.dump(self.training_history, f)
        
        # Save model card
        self._create_model_card()
        
        # Upload to Hub if requested
        if self.config.upload_to_hub and self.config.hf_token and self.config.hf_repo_name:
            self.upload_to_huggingface()
        
        if self.config.log_wandb:
            wandb.finish()
        
        print(f"Model saved to {self.config.output_dir}")
        return self.model, self.tokenizer
    
    def _create_model_card(self):
        """Create a model card markdown file for the fine-tuned model"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        model_card = f"""---
language: en
license: apache-2.0
tags:
- reinforcement-learning
- fine-tuned
- ppo
datasets:
- custom
---

# RL Fine-tuned {self.config.model_name.split('/')[-1]}

This model was fine-tuned from [{self.config.model_name}](https://huggingface.co/{self.config.model_name}) using Reinforcement Learning with PPO script made by [Spestly](https://github.com/Aayan-Mishra).

## Training procedure

The model was trained using a custom PPO implementation for {self.config.epochs} epochs with a learning rate of {self.config.learning_rate}.

### Training hyperparameters

- Learning rate: {self.config.learning_rate}
- Batch size: {self.config.batch_size}
- PPO epochs: {self.config.ppo_epochs}
- PPO clip parameter: {self.config.clip_param}
- Entropy coefficient: {self.config.entropy_coef}
- Number of training epochs: {self.config.epochs}

### Framework versions

- Transformers: {importlib.metadata.version('transformers')}
- PyTorch: {torch.__version__}

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "{self.config.hf_repo_name if self.config.hf_repo_name else 'path/to/model'}"

model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "What is reinforcement learning?"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Example outputs

{self._format_examples_for_model_card()}

## Training date
{timestamp}
"""
        
        with open(os.path.join(self.config.output_dir, "README.md"), "w") as f:
            f.write(model_card)
    
    def _format_examples_for_model_card(self):
        """Format evaluation examples for the model card"""
        examples = self.evaluate()
        formatted = ""
        
        for i, example in enumerate(examples, 1):
            formatted += f"### Example {i}\n\n"
            formatted += f"**Prompt:** {example['prompt']}\n\n"
            formatted += f"**Response:** {example['response']}\n\n"
        
        return formatted
    
    def upload_to_huggingface(self):
        """Upload the fine-tuned model to HuggingFace Hub"""
        print(f"Uploading model to HuggingFace Hub as {self.config.hf_repo_name}...")
        
        # Create repo if it doesn't exist
        api = HfApi()
        try:
            api.repo_info(
                repo_id=self.config.hf_repo_name,
                token=self.config.hf_token
            )
            print(f"Repository {self.config.hf_repo_name} already exists, will upload to it.")
        except Exception:
            print(f"Creating new repository: {self.config.hf_repo_name}")
            create_repo(
                repo_id=self.config.hf_repo_name,
                token=self.config.hf_token,
                private=False,
                exist_ok=True
            )
        
        # Create a temporary clone to push from
        repo_url = f"https://huggingface.co/{self.config.hf_repo_name}"
        tmp_dir = f"{self.config.output_dir}_hf_tmp"
        
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        
        # Clone or create new repo
        try:
            repo = Repository(
                local_dir=tmp_dir,
                clone_from=repo_url,
                token=self.config.hf_token
            )
        except Exception:
            # Create new repo locally
            os.makedirs(tmp_dir, exist_ok=True)
            repo = Repository(
                local_dir=tmp_dir,
                token=self.config.hf_token
            )
        
        # Copy model files to the repo directory
        for item in os.listdir(self.config.output_dir):
            source = os.path.join(self.config.output_dir, item)
            dest = os.path.join(tmp_dir, item)
            
            if os.path.isdir(source):
                if os.path.exists(dest):
                    shutil.rmtree(dest)
                shutil.copytree(source, dest)
            else:
                shutil.copy2(source, dest)
        
        # Push to Hub
        repo.push_to_hub(
            commit_message=f"Upload RL fine-tuned model from {self.config.model_name}"
        )
        
        print(f"Model uploaded to {repo_url}")
        return repo_url

# Command-line interface
def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a model using RL with PPO")
    
    parser.add_argument("--model_name", type=str, default="gpt2", help="Name or path of pre-trained model")
    parser.add_argument("--output_dir", type=str, default="rl_finetuned_model", help="Directory to save model")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--human_feedback", action="store_true", help="Use human feedback")
    parser.add_argument("--log_wandb", action="store_true", help="Log to Weights & Biases")
    parser.add_argument("--wandb_project", type=str, default="rl-finetuning", help="W&B project name")
    parser.add_argument("--upload_to_hub", action="store_true", help="Upload model to HuggingFace Hub")
    parser.add_argument("--hf_repo_name", type=str, default=None, help="HuggingFace repo name")
    parser.add_argument("--hf_token", type=str, default=None, help="HuggingFace token")
    parser.add_argument("--prompts_file", type=str, default=None, help="File containing training prompts (JSON)")
    
    return parser.parse_args()

# Example usage
def main():
    args = parse_args()
    
    # Load prompts from file if provided
    if args.prompts_file and os.path.exists(args.prompts_file):
        with open(args.prompts_file, "r") as f:
            prompts = json.load(f)
    else:
        # Example prompts for the model to respond to during RL training
        prompts = [
            "Write a concise summary of reinforcement learning:",
            "Explain the concept of reward in machine learning:",
            "What are the advantages of fine-tuning a model?",
            "How does PPO compare to other RL algorithms?",
            "Describe the process of training a language model:",
        ]
    
    # Create config from args
    config = RLConfig(
        model_name=args.model_name,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        human_feedback=args.human_feedback,
        log_wandb=args.log_wandb,
        wandb_project=args.wandb_project,
        output_dir=args.output_dir,
        upload_to_hub=args.upload_to_hub,
        hf_repo_name=args.hf_repo_name,
        hf_token=args.hf_token
    )
    
    # Initialize and train agent
    agent = PPOAgent(config)
    agent.train(prompts)
    
    # Test the fine-tuned model
    print("\n" + "="*50)
    print("Testing fine-tuned model:")
    test_prompt = "What is the key benefit of using reinforcement learning for fine-tuning?"
    response = agent.generate_response(test_prompt)
    print(f"Prompt: {test_prompt}")
    print(f"Response: {response}")
    
    if config.upload_to_hub:
        print(f"\nYou can find your model at: https://huggingface.co/{config.hf_repo_name}")

# Script to upload a pre-existing RL fine-tuned model
def upload_existing_model():
    parser = argparse.ArgumentParser(description="Upload an existing model to HuggingFace Hub")
    
    parser.add_argument("--model_dir", type=str, required=True, help="Directory containing the model")
    parser.add_argument("--hf_repo_name", type=str, required=True, help="HuggingFace repo name")
    parser.add_argument("--hf_token", type=str, required=True, help="HuggingFace token")
    
    args = parser.parse_args()
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    
    # Upload to Hub
    print(f"Uploading model to HuggingFace Hub as {args.hf_repo_name}...")
    
    api = HfApi()
    model.push_to_hub(args.hf_repo_name, token=args.hf_token)
    tokenizer.push_to_hub(args.hf_repo_name, token=args.hf_token)
    
    # Create a simple model card if it doesn't exist
    try:
        api.get_model_card_data(args.hf_repo_name, token=args.hf_token)
    except:
        readme_content = f"""---
language: en
license: apache-2.0
tags:
- reinforcement-learning
- fine-tuned
- ppo
---

# RL Fine-tuned Model

This model was fine-tuned using Reinforcement Learning with PPO.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{args.hf_repo_name}")
tokenizer = AutoTokenizer.from_pretrained("{args.hf_repo_name}")

prompt = "Your prompt here"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```
"""
        
        api.upload_file(
            path_or_fileobj=readme_content.encode(),
            path_in_repo="README.md",
            repo_id=args.hf_repo_name,
            token=args.hf_token
        )
    
    print(f"Model uploaded to https://huggingface.co/{args.hf_repo_name}")

if __name__ == "__main__":
    # Choose which script to run
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "upload_only":
        # Remove the first argument to allow argparse to work correctly
        sys.argv.pop(1)
        upload_existing_model()
    else:
        main()
