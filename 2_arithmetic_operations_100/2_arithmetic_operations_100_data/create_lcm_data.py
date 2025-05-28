import random
import pandas as pd
from transformer_lens import HookedTransformer

# Load the model and tokenizer
model_name = 'pythia-1.4b-deduped'
model = HookedTransformer.from_pretrained(model_name, device='cuda')
tokenizer = model.tokenizer

# Check if a number is a single token
def is_single_token(num):
    tokens = tokenizer.tokenize(str(num))
    return len(tokens) == 1

# Calculate the least common multiple (LCM)
def calculate_lcm(a, b):
    def gcd(x, y):
        while y:
            x, y = y, x % y
        return x
    return abs(a * b) // gcd(a, b)

# Generate data where one number is a multiple of the other
def generate_multiple_pair(max_base=50, max_multiplier=10):
    base = random.randint(1, max_base)
    multiplier = random.randint(2, max_multiplier)  # multiplier >= 2 to ensure num2 > num1
    num1 = base
    num2 = base * multiplier
    lcm = num2
    return num1, num2, lcm

# Generate data where two numbers are coprime
def generate_coprime_pair(max_num=100):
    while True:
        num1 = random.randint(1, max_num)
        num2 = random.randint(1, max_num)
        def gcd(x, y):
            while y:
                x, y = y, x % y
            return x
        if gcd(num1, num2) == 1:
            lcm = num1 * num2
            return num1, num2, lcm

# Generate data where two numbers share a common factor but are not multiples
def generate_common_factor_pair(max_base=50, max_factor=10):
    while True:
        base = random.randint(2, max_base)
        factor1 = random.randint(2, max_factor)
        factor2 = random.randint(2, max_factor)
        num1 = base * factor1
        num2 = base * factor2
        if num1 != num2:
            lcm = calculate_lcm(num1, num2)
            return num1, num2, lcm

# Main function to generate data
def generate_data(num_samples):
    data = []
    used_final_lcms = set()  # Track already generated (final_num1, final_num2) pairs

    # To ensure uniqueness, store (num1, num2) in sorted order
    def add_to_used(num1, num2):
        return tuple(sorted((num1, num2)))

    max_attempts = num_samples * 100  # Set a higher maximum number of attempts
    attempts = 0

    while len(data) < num_samples and attempts < max_attempts:
        attempts += 1
        choice = random.choice(["multiple", "coprime", "common_factor"])

        if choice == "multiple":
            generate_pair = generate_multiple_pair
        elif choice == "coprime":
            generate_pair = generate_coprime_pair
        else:
            generate_pair = generate_common_factor_pair

        # Generate a pair of numbers
        num1_1, num2_1, lcm_1 = generate_pair()

        # Check if lcm_1 is a single token
        if not is_single_token(lcm_1):
            continue  # Skip if lcm_1 is not a single token

        # Generate final_lcm
        for _ in range(20):  # Increase number of attempts
            final_num1, final_num2, final_lcm = generate_pair()
            sorted_final_pair = add_to_used(final_num1, final_num2)
            if sorted_final_pair not in used_final_lcms and is_single_token(final_lcm):
                used_final_lcms.add(sorted_final_pair)  # Record the generated final_lcm pair
                break
        else:
            continue  # Skip if no valid final_lcm is generated

        # Generate corrupt_lcm
        for _ in range(20):  # Increase number of attempts
            corrupt_num1, corrupt_num2, corrupt_lcm = generate_pair()
            if is_single_token(corrupt_lcm) and corrupt_lcm != final_lcm:
                break
        else:
            continue  # Skip if no valid corrupt_lcm is generated

        # Create the string representation for clean data
        clean_input = (
            f"Calculate the least common multiple (LCM) of two numbers. LCM({num1_1}, {num2_1}) = {lcm_1}, "
            f"LCM({final_num1}, {final_num2}) ="
        )

        # Create the string representation for corrupted data (semantically identical, only the final value changes)
        corrupt_input = (
            f"Calculate the least common multiple (LCM) of two numbers. LCM({num1_1}, {num2_1}) = {lcm_1}, "
            f"LCM({corrupt_num1}, {corrupt_num2}) ="
        )

        # Add the data to the list
        data.append({"clean": clean_input, "corrupted": corrupt_input, "label": final_lcm})

        # Print progress
        if len(data) % 100 == 0:
            print(f"Generated {len(data)}/{num_samples} samples...")

    if len(data) < num_samples:
        print(f"Could not generate enough data. Only generated {len(data)} samples.")
    else:
        print(f"Successfully generated {len(data)} samples.")

    return data

num_samples = 2500
data = generate_data(num_samples)

# Check the actual number of samples generated
actual_samples = len(data)
if actual_samples < num_samples:
    print(f"Could not generate enough data. Only generated {actual_samples} samples.")
else:
    print(f"Successfully generated {actual_samples} samples.")

# Convert to DataFrame
df = pd.DataFrame(data)

# Save as a CSV file
output_file = "lcm_reasoning_data.csv"
df.to_csv(output_file, index=False)

print(f"Data has been generated and saved to {output_file}")