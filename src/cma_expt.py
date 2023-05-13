import os
import torch
import random
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer

from cma import CMA

def prepare_sst2_data(sst2_dataset, tokenizer, device):
    # Preprocess data
    preprocessed_data = []
    for example in sst2_dataset:
        sentence = example["sentence"]
        label = example["label"]
        preprocessed_data.append({"sentence": sentence, "label": label})

    # Encode data
    encoded_data = []
    for example in tqdm(preprocessed_data, total=len(preprocessed_data)):
        sentence = example["sentence"]
        label = example["label"]

        intervention_sentence = sentence
        intervention_encoded = tokenizer(intervention_sentence, padding=True, return_tensors="pt").to(device)

        mask_token = tokenizer.mask_token if tokenizer.mask_token is not None else " "
        control_sentence = " ".join([mask_token for _ in range(intervention_encoded["input_ids"].shape[1] - 2)])
        control_encoded = tokenizer(control_sentence, padding=True, return_tensors="pt").to(device)

        inputs = {
            "control": control_encoded,
            "intervention": intervention_encoded
        }
        encoded_data.append(inputs)

    return encoded_data

def preprocess_and_encode_sst2_data(checkpoint, dataset, device):
    # Load dataset
    sst2_dataset = load_dataset("glue", "sst2")[dataset]

    # Set up tokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    # Prepare the SST-2 data
    encoded_data = prepare_sst2_data(sst2_dataset, tokenizer, device)

    return encoded_data

def run_cma_sst2_pipeline(checkpoint, outcome, encoded_data, num_samples, device):
    # Set up tokenizer, model, and CMA instance
    with torch.no_grad():
        torch.cuda.empty_cache()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    cma = CMA(checkpoint, outcome, device)

    # Set up parameters for causal mediation analysis
    random_sample = random.sample(encoded_data, num_samples)
    layers = range(len(cma.layers))

    meds_neuron = [[[i] for i in range(cma.num_neurons)] for _ in layers]

    # Calculate meds_token for each instance separately
    meds_token_list = [
        [[[i] for i in range(len(instance["intervention"]["input_ids"][0]))] for _ in layers]
        for instance in random_sample
    ]

    meds_list = {"neuron": meds_neuron, "token": meds_token_list}

    # Run causal mediation analysis
    effects = cma.get_sample_effects(
        random_sample, "MI", meds_list, num_batch=1, med_type="token", layers=layers
    )

    # Get token lists for each instance
    token_lists = []
    for instance in random_sample:
        tokens = tokenizer.convert_ids_to_tokens(instance["intervention"]["input_ids"][0])
        token_lists.append(tokens)

    return effects, token_lists

# Save effects and token_lists to files
def save_data(effects, token_lists, effects_file, token_lists_file):
    with open(effects_file, 'wb') as ef:
        pickle.dump(effects, ef)

    with open(token_lists_file, 'wb') as tlf:
        pickle.dump(token_lists, tlf)

# Load effects and token_lists from files
def load_data(effects_file, token_lists_file):
    with open(effects_file, 'rb') as ef:
        effects = pickle.load(ef)

    with open(token_lists_file, 'rb') as tlf:
        token_lists = pickle.load(tlf)

    return effects, token_lists
    
def plot_heatmaps(effects, token_lists, layer_indices=[0], instance_idx=100):
    # Get the tokens for the given instance
    tokens = token_lists[instance_idx]

    # Create a grid of subplots
    num_layers = len(layer_indices)
    ncols = 3
    nrows = (num_layers + ncols - 1) // ncols
    
    # Create a shared figure for all subplots
    sns.set_style("dark")
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * len(tokens), nrows * len(tokens)))

    for idx, layer_idx in enumerate(layer_indices):
        # Select the effects for the given layer and instance
        instance_effects = effects[instance_idx][layer_idx].copy()

        # Multiply diagonal values by -1
        np.fill_diagonal(instance_effects, -1 * np.diag(instance_effects))

        # Create a mask for the upper triangular part
        mask = np.triu(np.ones_like(instance_effects, dtype=bool), k=1)

        # Create a heatmap using seaborn
        ax = axes[idx // ncols, idx % ncols] if nrows > 1 else axes[idx % ncols]
        sns.heatmap(instance_effects, annot=True, fmt=".2f", xticklabels=tokens, yticklabels=tokens, cmap="coolwarm", center=0, cbar=False, mask=mask, square=True, ax=ax)
        ax.set_title(f"Layer {layer_idx}")

    # Remove unused subplots
    for idx in range(num_layers, nrows * ncols):
        ax = axes[idx // ncols, idx % ncols] if nrows > 1 else axes[idx % ncols]
        ax.axis("off")

    # Display the plots
    plt.tight_layout()
    plt.show()

def plot_avg_layer_effects(effects, thresholds=[0.1]):
    num_layers = len(effects[0])

    # Prepare data for Seaborn lineplot
    data = []
    data_diag = []
    for threshold in thresholds:
        for instance_idx, instance_effects in enumerate(effects):
            for layer_idx in range(num_layers):
                total_sum = np.sum(np.abs(np.tril(instance_effects[layer_idx])))
                layer_effect = np.sum(np.abs(np.tril(instance_effects[layer_idx], k=-1))[np.abs(np.tril(instance_effects[layer_idx], k=-1)) > threshold]) / total_sum
                layer_effect_diag = np.sum(np.abs(np.diag(instance_effects[layer_idx]))[np.abs(np.diag(instance_effects[layer_idx])) > threshold]) / total_sum
                data.append({"threshold": threshold, "layer": layer_idx, "effect": layer_effect, "instance": instance_idx})
                data_diag.append({"threshold": threshold, "layer": layer_idx, "effect": layer_effect_diag, "instance": instance_idx})

    data_df = pd.DataFrame(data)
    data_diag_df = pd.DataFrame(data_diag)

    # Plot multiple lines with different thresholds using Seaborn lineplot
    plt.figure(figsize=(20, 5))

    plt.subplot(1, 2, 1)
    ax = sns.lineplot(data=data_df, x="layer", y="effect", hue="threshold", marker="o", errorbar=('ci', 95))
    plt.xlabel("Layer")
    plt.ylabel("Normalized layer effect (MI)")
    ax.set_xticks(data_df['layer'].unique())  # Set xticks based on unique "layer" values in data_df

    plt.subplot(1, 2, 2)
    ax_diag = sns.lineplot(data=data_diag_df, x="layer", y="effect", hue="threshold", marker="o", errorbar=('ci', 95))
    plt.xlabel("Layer")
    plt.ylabel("Normalized layer effect (PIE)")
    ax_diag.set_xticks(data_diag_df['layer'].unique())  # Set xticks based on unique "layer" values in data_diag_df

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Set random seed
    random.seed(18)

    # Set device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Encode SST-2 data
    checkpoint = "assemblyai/bert-large-uncased-sst2"
    encoded_data = preprocess_and_encode_sst2_data(checkpoint, "validation", device=device)

    # Run causal mediation analysis
    effects_file = "../output/effects.pkl"
    token_lists_file = "../output/token_lists.pkl"
    if os.path.exists(effects_file) and os.path.exists(token_lists_file):
        effects, token_lists = load_data(effects_file, token_lists_file)
    else:
        outcome = "label_1"
        effects, token_lists = run_cma_sst2_pipeline(checkpoint, outcome, encoded_data, num_samples=len(encoded_data), device=device)
        save_data(effects, token_lists, effects_file, token_lists_file)

    # Plot heatmaps for the sentence 'The wild thornberrys movie is a jolly surprise'
    plot_heatmaps(effects, token_lists, layer_indices=[0, 12, 24], instance_idx=12)

    # Plot average layer effects
    plot_avg_layer_effects(effects, thresholds=[0, 0.2, 0.4, 0.6, 0.8, 1])