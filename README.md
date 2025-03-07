# Reinforcement Learning Fine-tuning for Language Models

This repository contains a Python implementation for fine-tuning pre-trained language models using Reinforcement Learning (RL) with Proximal Policy Optimization (PPO).

## Features

- Fine-tune any Hugging Face language model using RL
- Support for human feedback (RLHF) or automated reward models
- Configurable hyperparameters for PPO optimization
- Integrated with Weights & Biases for tracking experiments
- Automatic uploading to Hugging Face Hub
- Detailed model card generation

## Installation

```bash
git clone https://huggingface.co/Aayan-Mishra/Post-Training-RL
cd rl-finetuning
pip install -r requirements.txt
```

## Requirements

```
torch>=1.10.0
transformers>=4.18.0
tqdm
numpy
wandb
huggingface_hub
```

## Quick Start

### Training a model with RL

```bash
python train_rl.py \
  --model_name=gpt2 \
  --output_dir=rl_finetuned_model \
  --epochs=5 \
  --batch_size=4 \
  --learning_rate=2e-5 \
  --upload_to_hub \
  --hf_repo_name=YOUR_USERNAME/my-rl-model \
  --hf_token=YOUR_HF_TOKEN
```

### Using human feedback

```bash
python train_rl.py \
  --model_name=gpt2 \
  --human_feedback \
  --epochs=3 \
  --batch_size=2
```

### Uploading an existing model

```bash
python train_rl.py upload_only \
  --model_dir=./my_model_directory \
  --hf_repo_name=YOUR_USERNAME/my-rl-model \
  --hf_token=YOUR_HF_TOKEN
```

## How It Works

1. **Initial setup**: Load a pre-trained model and tokenizer
2. **Sample generation**: Generate responses to prompts using the current policy
3. **Reward calculation**: Calculate rewards using either:
   - Human feedback (interactive rating)
   - A separate reward model
   - Heuristic rules
4. **Policy update**: Update the model using PPO to maximize rewards while preventing too large policy changes
5. **Evaluation**: Test the fine-tuned model on held-out prompts
6. **Upload**: Push the model to Hugging Face Hub with model card and examples

## Example

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load your fine-tuned model
model_name = "YOUR_USERNAME/my-rl-model"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Use the model
prompt = "What is reinforcement learning?"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Customizing Rewards

You can customize the reward function by modifying the `get_reward` method in the `RewardModel` class. The current implementation supports:

1. Human feedback (interactive rating 0-10)
2. Using a separate reward model
3. Simple heuristics based on response length and repetition

To implement a custom reward model, create a new class that inherits from `RewardModel` and override the `get_reward` method.

## Citation

If you use this code in your research or project, please cite this repository:

```
@software{rl_finetuning,
  author = {Aayan Mishra},
  title = {Reinforcement Learning Fine-tuning for Language Models},
  url = {https://huggingface.co/Spestly},
  year = {2025},
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
