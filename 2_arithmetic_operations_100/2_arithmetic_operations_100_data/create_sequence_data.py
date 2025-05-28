import random
import pandas as pd
from transformer_lens import HookedTransformer
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Load the model and tokenizer
model_name = 'gpt-neo-2.7B'
model = HookedTransformer.from_pretrained(model_name, device='cuda')
tokenizer = model.tokenizer

# Check if a number is a single token
def is_single_token(num):
    tokens = tokenizer.tokenize(str(num))
    return len(tokens) == 1

# Generate an arithmetic or geometric sequence, ensuring each number in the sequence is a single token
def generate_arithmetic_or_geometric_sequence():
    sequence_type = random.choice(["arithmetic", "geometric"])  # Randomly choose the type of sequence

    if sequence_type == "arithmetic":
        # Arithmetic sequence: starting value and step
        start = random.randint(1, 500)
        step = random.randint(1, 100)
        sequence = [start + i * step for i in range(5)]  # Length of 5
    else:
        # Geometric sequence: starting value and ratio
        start = random.randint(1, 10)
        step = random.randint(2, 10)
        sequence = [start * (step ** i) for i in range(5)]  # Length of 5

    # Ensure each number in the sequence is a single token
    if not all(is_single_token(num) for num in sequence):
        return None, None

    # Extract the last number as the label
    label = sequence[-1]

    # Extract the first four numbers as the sequence
    sequence = sequence[:-1]

    return sequence, label

# Generate corrupt data
def corrupt_sequence(sequence):
    corrupted_sequence = sequence[:]
    index_to_corrupt = random.choice([-1, -2])  # Only perturb one of the last two numbers
    corrupted_value = corrupted_sequence[index_to_corrupt]

    # Add perturbation
    while True:
        new_value = corrupted_value + random.randint(1, 100)  # Add perturbation
        if is_single_token(new_value) and new_value != corrupted_value and new_value not in corrupted_sequence:
            corrupted_sequence[index_to_corrupt] = new_value
            break

    return corrupted_sequence

# Check if the token sequence lengths of clean and corrupted texts are the same
def are_token_lengths_equal(clean, corrupted):
    clean_tokens = tokenizer.tokenize(clean)
    corrupted_tokens = tokenizer.tokenize(corrupted)
    return len(clean_tokens) == len(corrupted_tokens)

# Main function to generate data
def generate_data(num_samples):
    data = []
    seen_sequences = set()  # Used to check for duplicate sequences

    while len(data) < num_samples:
        # Generate clean data
        sequence, label = generate_arithmetic_or_geometric_sequence()

        if sequence is None or label is None:
            continue  # Skip if sequence generation fails

        # Convert the sequence to a tuple to check for duplicates
        sequence_tuple = tuple(sequence)
        if sequence_tuple in seen_sequences:
            continue  # Skip if the sequence is duplicate

        seen_sequences.add(sequence_tuple)  # Record the new sequence

        # Create the string representation for clean data
        clean_input = f"Derive the following sequence: {', '.join(map(str, sequence))},"
        
        # Generate corrupt data
        corrupted_sequence = corrupt_sequence(sequence)
        corrupt_input = f"Derive the following sequence: {', '.join(map(str, corrupted_sequence))},"

        # Check if the token sequence lengths of clean and corrupted texts are the same
        if not are_token_lengths_equal(clean_input, corrupt_input):
            continue  # Skip if lengths are not the same

        # Add the data to the list
        data.append({"clean": clean_input, "corrupted": corrupt_input, "label": label})

    return data

# Generate 5000 data points
num_samples = 5000
data = generate_data(num_samples)

# Convert to DataFrame
df = pd.DataFrame(data)

# Save as a CSV file
output_file = "sequence_reasoning_gpt_data.csv"
df.to_csv(output_file, index=False)

print(f"Data has been generated and saved to {output_file}")