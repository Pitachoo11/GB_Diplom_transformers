import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
import random
import os
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler
import time
import logging

# Configure the logging settings
log_dir = "./logs"  # Adjust this path as needed
os.makedirs(log_dir, exist_ok=True)

log_file = os.path.join(log_dir, "training_log.txt")
logging.basicConfig(
    filename=log_file,
    format="%(asctime)s [%(levelname)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO
)

# Add a console handler to display logs on the console
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter("%(message)s"))
logging.getLogger().addHandler(console_handler)

# Create a TensorBoard SummaryWriter
writer = SummaryWriter(log_dir=log_dir)

# Define the parameter sets
parameter_sets = {
    "Parameter Set #1 (Balanced)": {
        "batch_size": 64,
        "block_size": 256,
        "learning_rate": 3e-4,
        "n_layer": 6,
        "n_head": 6,
        "dropout": 0.2,
        "data_augmentation": True,
        "early_stopping": {"patience": 3, "min_delta": 0.001},
        "log_dir": "./logs",
        "checkpoint_dir": "./checkpoints",
    },
    "Parameter Set #2 (Faster Training)": {
        "batch_size": 128,
        "block_size": 128,
        "learning_rate": 1e-3,
        "n_layer": 4,
        "n_head": 4,
        "dropout": 0.3,
        "data_augmentation": False,
        "early_stopping": {"patience": 5, "min_delta": 0.001},
        "log_dir": "./logs",
        "checkpoint_dir": "./checkpoints",
    },
    "Parameter Set #3 (Precise)": {
        "batch_size": 16,                # Smaller batch size for precision
        "block_size": 512,               # Larger block size for context
        "learning_rate": 1e-4,          # Lower learning rate for stability
        "n_layer": 12,                  # Deeper model for quality
        "n_head": 8,                   # More attention heads for better attention
        "dropout": 0.1,                 # Lower dropout for more information retention
        "data_augmentation": True,       # Use data augmentation for better generalization
        "early_stopping": {"patience": 10, "min_delta": 0.0001},  # Be patient and strict with early stopping
        "log_dir": "./logs",
        "checkpoint_dir": "./checkpoints",
    },
}

# Display the menu and let the user choose a parameter set
print("Choose a Parameter Set:")
for i, param_set_name in enumerate(parameter_sets.keys(), start=1):
    print(f"{i}. {param_set_name}")

while True:
    try:
        choice = int(input("Enter the number of the parameter set: "))
        if 1 <= choice <= len(parameter_sets):
            selected_param_set = list(parameter_sets.keys())[choice - 1]
            params = parameter_sets[selected_param_set]
            break
        else:
            print("Invalid choice. Please enter a valid number.")
    except ValueError:
        print("Invalid input. Please enter a number.")

# Configuration dictionary for hyperparameters and settings
config = {
    "hyperparameters": {
        "batch_size": params["batch_size"],
        "block_size": params["block_size"],
        "max_iters": 10000,
        "eval_interval": 500,
        "learning_rate": params["learning_rate"],
        "device": 'cuda' if torch.cuda.is_available() else 'cpu',
        "eval_iters": 10,
        "n_embd": 512,
        "n_head": params["n_head"],
        "n_layer": params["n_layer"],
        "dropout": params["dropout"],
        "lr_schedule": {
            "step_size": 1000,
            "gamma": 0.5,
        },
        "grad_clip": 0.25,
    },
    "data": {
        "file_path": "/content/v2_input_dataset.txt",
        "split_ratio": 0.8,
    },
    "checkpoint": {
        "save_dir": params["checkpoint_dir"],
        "load_path": None,
    },
    "early_stopping": params["early_stopping"],
    "data_augmentation": {
        "shuffle_within_batch": params["data_augmentation"],
    },
    "logging": {
        "log_dir": params["log_dir"],
    }
}

# Custom Dataset class
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, text, block_size):
        self.block_size = block_size
        self.examples = [text[i:i+block_size] for i in range(0, len(text)-block_size+1)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

# Custom collate function for DataLoader
def collate_fn(batch):
    return torch.tensor(batch)

# Load text data
def load_text_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text

# Split data into train and validation
def split_train_validation_data(text, split_ratio):
    n = int(split_ratio * len(text))
    train_data = text[:n]
    val_data = text[n:]
    return train_data, val_data

# Load and preprocess text data
text = load_text_data(config["data"]["file_path"])

# Create a character-to-integer mapping
vocab = set(text)
stoi = {char: i for i, char in enumerate(vocab)}

# Compute the vocabulary size
vocab_size = len(vocab)

# Convert text to integers using a mapping
def encode(text):
    return [stoi[c] for c in text]

# Define the 'decode' function to convert integers back to characters
def decode(int_tokens):
    return ''.join([list(vocab)[i] for i in int_tokens])

# Estimate loss on train and validation data
@torch.no_grad()
def estimate_loss(model, eval_iters, get_batch_fn):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch_fn(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Generate text from the model
def generate_text(model, context, max_new_tokens):
    for _ in range(max_new_tokens):
        # crop context to the last block_size tokens
        context_cond = context[:, -config["hyperparameters"]["block_size"]:]
        # get the predictions
        logits, _ = model(context_cond)
        # focus only on the last time step
        logits = logits[:, -1, :]  # becomes (B, C)
        # apply softmax to get probabilities
        probs = F.softmax(logits, dim=-1)  # (B, C)
        # sample from the distribution
        next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
        # append sampled token to the running sequence
        context = torch.cat((context, next_token), dim=1)  # (B, T+1)
    return context

# Define the Bigram Language Model
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, config["hyperparameters"]["n_embd"])
        self.position_embedding_table = nn.Embedding(config["hyperparameters"]["block_size"], config["hyperparameters"]["n_embd"])
        self.blocks = nn.Sequential(*[Block(config["hyperparameters"]["n_embd"], n_head=config["hyperparameters"]["n_head"]) for _ in range(config["hyperparameters"]["n_layer"])])
        self.ln_f = nn.LayerNorm(config["hyperparameters"]["n_embd"])
        self.lm_head = nn.Linear(config["hyperparameters"]["n_embd"], vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=config["hyperparameters"]["device"]))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

# Define a function to get a batch of data
def get_batch(split):
    data = train_data if split == 'train' else val_data
    data = torch.tensor(encode(data), dtype=torch.long)  # Convert data to a tensor
    ix = torch.randint(len(data) - config["hyperparameters"]["block_size"], (config["hyperparameters"]["batch_size"],))
    x = torch.stack([data[i:i+config["hyperparameters"]["block_size"]] for i in ix])
    y = torch.stack([data[i+1:i+config["hyperparameters"]["block_size"]+1] for i in ix])
    x, y = x.to(config["hyperparameters"]["device"]), y.to(config["hyperparameters"]["device"])
    return x, y

# Define the Head class for self-attention
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(config["hyperparameters"]["n_embd"], head_size, bias=False)
        self.query = nn.Linear(config["hyperparameters"]["n_embd"], head_size, bias=False)
        self.value = nn.Linear(config["hyperparameters"]["n_embd"], head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(config["hyperparameters"]["block_size"], config["hyperparameters"]["block_size"])))
        self.dropout = nn.Dropout(config["hyperparameters"]["dropout"])

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

# Define the MultiHeadAttention class
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(config["hyperparameters"]["n_embd"], config["hyperparameters"]["n_embd"])
        self.dropout = nn.Dropout(config["hyperparameters"]["dropout"])

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

# Define the FeedForward class
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(config["hyperparameters"]["dropout"]),
        )

    def forward(self, x):
        return self.net(x)

# Define the Block class for the Transformer
class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# Set a random seed for reproducibility
torch.manual_seed(1337)

# Load and preprocess text data
text = load_text_data(config["data"]["file_path"])
train_data, val_data = split_train_validation_data(text, config["data"]["split_ratio"])

# Convert train_data and val_data to tensors
train_data = torch.tensor(encode(train_data), dtype=torch.long)
val_data = torch.tensor(encode(val_data), dtype=torch.long)

# Split data into train and validation sets
train_data, val_data = split_train_validation_data(text, config["data"]["split_ratio"])

# Create custom datasets and data loaders
train_dataset = TextDataset(train_data, config["hyperparameters"]["block_size"])
val_dataset = TextDataset(val_data, config["hyperparameters"]["block_size"])

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=config["hyperparameters"]["batch_size"],
    shuffle=True,
    collate_fn=collate_fn
)

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=config["hyperparameters"]["batch_size"],
    shuffle=False,
    collate_fn=collate_fn
)

# Create the Bigram Language Model
model = BigramLanguageModel().to(config["hyperparameters"]["device"])

# Print the number of model parameters
print(sum(p.numel() for p in model.parameters()) / 1e6, 'M parameters')

# Specify the directory for saving checkpoints
if config["checkpoint"]["save_dir"]:
    checkpoint_dir = config["checkpoint"]["save_dir"]
    checkpoint_name = f"model_checkpoint_{iter}.pth"

    # Create the directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

    # Save the model checkpoint
    torch.save(model.state_dict(), checkpoint_path)

# Create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=config["hyperparameters"]["learning_rate"])

# Learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=config["hyperparameters"]["lr_schedule"]["step_size"],
    gamma=config["hyperparameters"]["lr_schedule"]["gamma"]
)

# Initialize GradScaler for mixed-precision training
scaler = GradScaler()

# Initialize variables for tracking best validation loss
best_val_loss = float("inf")
no_improvement = 0

# Training loop
for iter in tqdm(range(config["hyperparameters"]["max_iters"]), desc="Training Iterations"):
    # Adjust the learning rate
    lr_scheduler.step()

    if iter % config["hyperparameters"]["eval_interval"] == 0 or iter == config["hyperparameters"]["max_iters"] - 1:
        losses = estimate_loss(model, config["hyperparameters"]["eval_iters"], get_batch)
        logging.info(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # Log metrics to TensorBoard
        writer.add_scalar("Train Loss", losses['train'], iter)
        writer.add_scalar("Validation Loss", losses['val'], iter)

        if losses['val'] + config["early_stopping"]["min_delta"] < best_val_loss:
            best_val_loss = losses['val']
            if config["checkpoint"]["save_dir"]:
                # Save model checkpoint with a unique name based on the current iteration
                checkpoint_name = f"model_checkpoint_{iter}.pth"
                checkpoint_path = os.path.join(config["checkpoint"]["save_dir"], checkpoint_name)
                torch.save(model.state_dict(), checkpoint_path)

                # Generate and print sample text
                context = torch.zeros((1, 1), dtype=torch.long, device=config["hyperparameters"]["device"])
                generated_text = generate_text(model, context, max_new_tokens=500)
                decoded_text = decode(generated_text[0].tolist())
                print("\nGenerated Text:")

                # Print the text character by character with a delay
                for char in decoded_text:
                    print(char, end='', flush=True)  # Use flush to immediately print the character
                    time.sleep(0.03)  # Adjust the sleep duration as needed

                print()

            no_improvement = 0
        else:
            no_improvement += 1

        if no_improvement >= config["early_stopping"]["patience"]:
            logging.info(f"Early stopping after {no_improvement} consecutive epochs without improvement.")
            break

    # Sample a batch of data
    X, Y = get_batch('train')

    # Evaluate the loss
    logits, loss = model(X, Y)

    # Zero the gradients
    optimizer.zero_grad(set_to_none=True)

    # Perform backpropagation with gradient scaling
    scaler.scale(loss).backward()

    # Update the model weights
    scaler.step(optimizer)

    # Update the GradScaler for the next iteration
    scaler.update()

    # Call optimizer.step() before lr_scheduler.step()
    optimizer.step()

# Close the TensorBoard SummaryWriter
writer.close()