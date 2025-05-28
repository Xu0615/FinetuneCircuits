import random
import pandas as pd
from transformer_lens import HookedTransformer

# Load the model and tokenizer
model_name = 'gpt-neo-2.7B'
model = HookedTransformer.from_pretrained(model_name, device='cuda')
tokenizer = model.tokenizer

# The operator list includes only multiplication and division
operators = ['*', '/']

# Generate random multiplication and division expressions
def generate_expression(operator=None):
    if operator is None:
        operator = random.choice(operators)

    if operator == '*':
        num1 = random.randint(2, 200)
        num2 = random.randint(2, 180)
    elif operator == '/':
        num2 = random.randint(2, 200)
        x = random.randint(2, 200)
        num1 = num2 * x  # Ensure num1 is divisible by num2

    expression = f"{num1} {operator} {num2}"
    return expression

# Corruption rules for multiplication and division
def corrupt_expression(clean_expr):
    num1, operator, num2 = clean_expr.split()
    num1, num2 = int(num1), int(num2)

    if operator == '*':
        new_num1 = random.choice([i for i in range(2, 100) if i != num1])
        corrupted_expr = f"{new_num1} {operator} {num2}"
    elif operator == '/':
        divisible_numbers = [i for i in range(num2, 200, num2) if i != num1 and is_single_token(i)]
        if not divisible_numbers:  # If the list is empty
            while True:
                x = random.randint(2, 200)  # Generate a new x
                new_num1 = num2 * x
                if is_single_token(new_num1):
                    break
        else:
            new_num1 = random.choice(divisible_numbers)
        corrupted_expr = f"{new_num1} {operator} {num2}"

    return corrupted_expr

# Compute the correct result of a clean expression
def calculate_result(expr):
    num1, operator, num2 = expr.split()
    num1, num2 = int(num1), int(num2)

    if operator == '*':
        return num1 * num2
    elif operator == '/':
        return num1 // num2  # Use integer division

# Check if the label is a single token after tokenization
def is_single_token(label):
    tokens = tokenizer.tokenize(str(label))
    return len(tokens) == 1

# Check if the token counts of the entire clean and corrupt strings are equal
def check_tokens_equal(full_clean_str, full_corrupt_str):
    clean_tokens = tokenizer.tokenize(full_clean_str)
    corrupted_tokens = tokenizer.tokenize(full_corrupt_str)
    return len(clean_tokens) == len(corrupted_tokens)

# Generate dataset
data = []
operation_count = {op: 0 for op in operators}
total_count = 2000  # Total number of data points
target_per_operation = total_count // len(operators)

max_attempts = total_count * 10
attempts = 0

unique_expressions = set()

instruction = "Calculate the result of the following arithmetic expression and provide only the final answer:"

# Prioritize generating data for unfinished operation categories
while sum(operation_count.values()) < total_count and attempts < max_attempts:
    attempts += 1
    for operator in operators:
        if operation_count[operator] < target_per_operation:
            clean_expr = generate_expression(operator=operator)
            if clean_expr not in unique_expressions:
                corrupted_expr = corrupt_expression(clean_expr)
                label = calculate_result(clean_expr)

                # Construct the full string
                clean_data = f"{instruction} {clean_expr} ="
                corrupted_data = f"{instruction} {corrupted_expr} ="
                corrupted_label = calculate_result(corrupted_expr)

                # Check if the results of clean and corrupt expressions are single tokens,
                # and if their token counts are equal
                if (is_single_token(label) and 
                    is_single_token(corrupted_label) and
                    is_single_token(int(clean_expr.split()[0])) and
                    check_tokens_equal(clean_data, corrupted_data)):
                    
                    data.append({
                        'clean': clean_data,
                        'corrupted': corrupted_data,
                        'label': label
                    })
                    unique_expressions.add(clean_expr)
                    operation_count[operator] += 1
                else:
                    # If conditions aren't met, you can choose to debug here
                    pass  # Or use a print statement for debugging

# If the data is insufficient, notify the user
if sum(operation_count.values()) < total_count:
    print(f"Only generated {sum(operation_count.values())} data points, unable to meet the requirements. Please adjust the number ranges.")

# Convert to DataFrame
df = pd.DataFrame(data)

# Save as CSV
df.to_csv("Logical_Operations_Mul_Div_gpt.csv", index=False)

# Display the number of data points for each operator
print("Number of data points for each operator:")
for op in operators:
    print(f"{op}: {operation_count[op]}")

# Additional validation: Check if all data points have consistent token counts
def verify_token_consistency(df):
    for idx, row in df.iterrows():
        clean_tokens = tokenizer.tokenize(row['clean'])
        corrupted_tokens = tokenizer.tokenize(row['corrupted'])
        if len(clean_tokens) != len(corrupted_tokens):
            print(f"Token count mismatch at index {idx}:")
            print(f"Clean: {row['clean']} ({len(clean_tokens)} tokens)")
            print(f"Corrupted: {row['corrupted']} ({len(corrupted_tokens)} tokens)")
            return False
    return True

if verify_token_consistency(df):
    print("All data points have consistent token counts.")
else:
    print("There are data points with inconsistent token counts.")