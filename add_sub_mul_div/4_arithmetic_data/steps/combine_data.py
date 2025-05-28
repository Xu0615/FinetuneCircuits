import random
import pandas as pd
from transformer_lens import HookedTransformer

# ---------- 1. Load model and tokenizer for token checking ----------
model_name = 'pythia-1.4b-deduped'
model = HookedTransformer.from_pretrained(model_name, device='cuda')
tokenizer = model.tokenizer

# ---------- 2. Set hyperparameters ----------
TOTAL_SAMPLES = 5000  # How many multi-step expression data to generate
MAX_ATTEMPTS = TOTAL_SAMPLES * 10  # Maximum attempts to prevent infinite loops
INSTRUCTION = "Calculate the result of the following arithmetic expression and provide only the final answer:"

# ---------- 3. Define operator combinations ----------
BRACKET_OPS = ['+', '-']   # Operators inside brackets
OUTSIDE_OPS = ['*', '/']   # Operators outside brackets

# ---------- 4. Utility functions ----------

def is_single_token(value) -> bool:
    """
    Check if an integer or string is tokenized as a single token in the tokenizer.
    If you don't need this restriction, you can directly return True.
    """
    tokens = tokenizer.tokenize(str(value))
    return len(tokens) == 1

def compute_bracket_result(x, bracket_op, y):
    """
    Calculate the result inside brackets: x ± y
    """
    if bracket_op == '+':
        return x + y
    elif bracket_op == '-':
        return x - y
    else:
        raise ValueError(f"Unknown bracket operator: {bracket_op}")

def compute_final_result(bracket_val, outside_op, z):
    """
    Perform * or / operation based on the bracket result
    """
    if outside_op == '*':
        return bracket_val * z
    elif outside_op == '/':
        # Use integer division here
        return bracket_val // z
    else:
        raise ValueError(f"Unknown outside operator: {outside_op}")

def parse_expression(expr: str):
    """
    Parse x, bracket_op, y, outside_op, z from a string like '(x ± y) ○ z'.
    Assume the format is standard, e.g., '(3 + 5) * 2'
    """
    parts = expr.replace('(', '').replace(')', '').split()
    x_str, bracket_op, y_str, outside_op, z_str = parts
    return int(x_str), bracket_op, int(y_str), outside_op, int(z_str)

def generate_clean_expression(target_op=None):
    """
    Randomly generate a multi-step arithmetic expression (x ± y) ○ z, ensuring:
      - If it's subtraction, x >= y
      - If it's outside division, the bracket result can be divided by z and z != 1
      - Can specify the outside operator
    Returns a string like '(3 + 5) * 2' and the correct result
    """
    bracket_op = random.choice(BRACKET_OPS)
    outside_op = target_op if target_op else random.choice(OUTSIDE_OPS)

    # Generate x, y
    x = random.randint(2, 100)
    y = random.randint(1, 100)
    if bracket_op == '-':
        if y > x:
            x, y = y, x

    bracket_val = compute_bracket_result(x, bracket_op, y)

    if bracket_val == 0 and outside_op == '/':
        return None, None

    if outside_op == '*':
        z = random.randint(2, 100)
    else:  # outside_op == '/'
        abs_val = abs(bracket_val)
        if abs_val <= 1:
            return None, None
        factors = [i for i in range(2, abs_val+1) if abs_val % i == 0]
        if not factors:
            return None, None
        z = random.choice(factors)

    expr_str = f"({x} {bracket_op} {y}) {outside_op} {z}"
    final_val = compute_final_result(bracket_val, outside_op, z)
    return expr_str, final_val

def corrupt_expression(clean_expr):
    """
    Only perturb the outside multiplication/division number z, keep the bracket part (x ± y) unchanged.
    """
    x, bracket_op, y, outside_op, z = parse_expression(clean_expr)
    bracket_val = compute_bracket_result(x, bracket_op, y)

    if outside_op == '*':
        while True:
            new_z = random.randint(2, 100)
            if new_z != z:
                break
        corrupt_expr = f"({x} {bracket_op} {y}) {outside_op} {new_z}"

    elif outside_op == '/':
        abs_val = abs(bracket_val)
        if abs_val <= 1:
            return None
        factors = [i for i in range(2, abs_val+1) if abs_val % i == 0]
        possible_divisors = [d for d in factors if d != z]
        if not possible_divisors:
            return None

        new_z = random.choice(possible_divisors)
        corrupt_expr = f"({x} {bracket_op} {y}) {outside_op} {new_z}"
    else:
        raise ValueError(f"Unknown outside operator: {outside_op}")

    return corrupt_expr

def calculate_result(expr_str):
    """
    Given a string like '(x ± y) ○ z', calculate the final result
    """
    x, bracket_op, y, outside_op, z = parse_expression(expr_str)
    bracket_val = compute_bracket_result(x, bracket_op, y)
    return compute_final_result(bracket_val, outside_op, z)

# ---------- 5. Main logic: generate data and build DataFrame ----------
def main():
    data = []
    unique_expressions = set()
    attempts = 0
    multiply_count = 0
    divide_count = 0

    while len(data) < TOTAL_SAMPLES and attempts < MAX_ATTEMPTS:
        attempts += 1

        # Control the generation ratio of multiplication and division
        if multiply_count <= divide_count:
            target_op = '*'
        else:
            target_op = '/'

        clean_expr, clean_val = generate_clean_expression(target_op)
        if clean_expr is None or clean_val is None:
            continue

        if clean_expr in unique_expressions:
            continue

        if not is_single_token(clean_val):
            continue

        corrupted_expr = corrupt_expression(clean_expr)
        if corrupted_expr is None:
            continue

        x, bracket_op, y, outside_op, z = parse_expression(clean_expr)
        if outside_op == '*':
            multiply_count += 1
        elif outside_op == '/':
            divide_count += 1

        clean_full_str = f"{INSTRUCTION} {clean_expr} ="
        corrupt_full_str = f"{INSTRUCTION} {corrupted_expr} ="
        if len(tokenizer.tokenize(clean_full_str)) != len(tokenizer.tokenize(corrupt_full_str)):
            continue

        data.append({
            'clean': f"{INSTRUCTION} {clean_expr} =",
            'corrupted': f"{INSTRUCTION} {corrupted_expr} =",
            'label': clean_val
        })
        unique_expressions.add(clean_expr)

    if len(data) < TOTAL_SAMPLES:
        print(f"Only generated {len(data)} data points, cannot meet the target of {TOTAL_SAMPLES}. Consider expanding the range or relaxing constraints.")

    df = pd.DataFrame(data)
    df.to_csv("Combined_Arithmetic.csv", index=False)
    print(f"Generated {len(data)} multi-step expressions and written to Combined_Arithmetic.csv")
    print(f"Number of expressions with multiplication outside brackets: {multiply_count}")
    print(f"Number of expressions with division outside brackets: {divide_count}")

if __name__ == "__main__":
    main()
