import random
import pandas as pd
from transformer_lens import HookedTransformer

model_name = 'pythia-1.4b-deduped'
model = HookedTransformer.from_pretrained(model_name, device='cuda')
tokenizer = model.tokenizer


def is_single_token(num):
    tokens = tokenizer.tokenize(str(num))
    return len(tokens) == 1

def generate_clean_data():
    a = random.randint(1, 20) 
    b = random.randint(1, 500)  

    y_values = [a * x + b for x in range(1, 5)] 

    # Check if all y values are single tokens
    if not all(is_single_token(y) for y in y_values):
        return None

    # Extract y1, y2, y3, and y4
    y1, y2, y3, y4 = y_values

    return {
        'a': a,
        'b': b,
        'y_values': [y1, y2, y3],
        'label': y4
    }

# Function to generate corrupt data
def generate_corrupt_data(a, original_b):
    # Select a different b'
    possible_b = list(range(1, 501))
    possible_b.remove(original_b)
    b_prime = random.choice(possible_b)

    # Compute y1', y2', y3', and y4'
    y_values_prime = [a * x + b_prime for x in range(1, 5)]

    # Check if all y' values are single tokens
    if not all(is_single_token(y) for y in y_values_prime):
        return None

    return {
        'b_prime': b_prime,
        'y_values_prime': [y_values_prime[0], y_values_prime[1], y_values_prime[2]],
        'label_prime': y_values_prime[3]
    }

# Check if the token sequence lengths of clean and corrupted inputs are the same
def are_token_lengths_equal(clean, corrupt):
    clean_tokens = tokenizer.tokenize(clean)
    corrupt_tokens = tokenizer.tokenize(corrupt)
    return len(clean_tokens) == len(corrupt_tokens)

# Main function to generate data
def generate_data(num_samples, max_total_attempts=100000):
    data = []
    seen_clean_b_pairs = set()  # Used to check for duplicate (a, b) pairs
    total_attempts = 0

    while len(data) < num_samples and total_attempts < max_total_attempts:
        total_attempts += 1

        # Generate clean data
        clean = generate_clean_data()
        if clean is None:
            continue  # Skip if clean data generation fails

        a = clean['a']
        b = clean['b']
        y_values = clean['y_values']  # [y1, y2, y3]
        label = clean['label']  # y4

        # Check for duplicate (a, b) pairs
        if (a, b) in seen_clean_b_pairs:
            continue
        seen_clean_b_pairs.add((a, b))

        # Generate corrupt data
        corrupt = generate_corrupt_data(a, b)
        if corrupt is None:
            continue  # Skip if corrupt data generation fails

        b_prime = corrupt['b_prime']
        y_values_prime = corrupt['y_values_prime']  # [y1', y2', y3']
        label_prime = corrupt['label_prime']  # y4'

        # Create string representations for clean and corrupt data
        clean_input = f"There is a function y={a}x+{b}. Given x=1,2,3,4, y={y_values[0]},{y_values[1]},{y_values[2]},"
        corrupt_input = f"There is a function y={a}x+{b_prime}. Given x=1,2,3,4, y={y_values_prime[0]},{y_values_prime[1]},{y_values_prime[2]},"

        # Check if token sequence lengths of clean and corrupted inputs are the same
        if not are_token_lengths_equal(clean_input, corrupt_input):
            continue  # Skip if lengths are not equal

        # Add data to the list
        data.append({
            "clean": clean_input,
            "corrupted": corrupt_input,
            "label": label  # Use clean y4 as the label
        })

    if len(data) < num_samples:
        print(f"Warning: Only generated {len(data)} samples, target was {num_samples}.")

    return data

if __name__ == "__main__":
    # Generate 5000 samples
    num_samples = 5000
    data = generate_data(num_samples)

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Save as CSV file
    output_file = "function_data.csv"
    df.to_csv(output_file, index=False)

    print(f"Data has been generated and saved to {output_file}, total samples: {len(data)}")