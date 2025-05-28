import random
import pandas as pd
from transformer_lens import HookedTransformer

model_name = 'pythia-1.4b-deduped'
model = HookedTransformer.from_pretrained(model_name, device='cuda')
tokenizer = model.tokenizer

operators = ['+', '-']

def generate_expression():
    operator = random.choice(operators)
    num1 = random.randint(1, 100)
    num2 = random.randint(1, 100)
    
    if num1 < num2:
        num1, num2 = num2, num1
    
    expression = f"{num1} {operator} {num2}"
    return expression

def corrupt_expression(clean_expr):
    num1, operator, num2 = clean_expr.split()
    
    new_operator = random.choice([op for op in operators if op != operator])
    corrupted_expr = f"{num1} {new_operator} {num2}"
    
    return corrupted_expr

def calculate_result(expr):
    num1, operator, num2 = expr.split()
    num1, num2 = int(num1), int(num2)
    
    if operator == '+':
        return num1 + num2
    elif operator == '-':
        return num1 - num2

def is_single_token(label):
    tokens = tokenizer.tokenize(str(label))
    return len(tokens) == 1

data = []
operation_count = {op: 0 for op in operators}
total_count = 5000  
target_per_operation = total_count // len(operators)

max_attempts = total_count * 10 
attempts = 0

unique_expressions = set()

instruction = ""

while sum(operation_count.values()) < total_count and attempts < max_attempts:
    attempts += 1
    clean_expr = generate_expression()
    operator = clean_expr.split()[1]
    
    if clean_expr not in unique_expressions and operation_count[operator] < target_per_operation:
        corrupted_expr = corrupt_expression(clean_expr)
        label = calculate_result(clean_expr)
        
        if is_single_token(label):
            data.append({
                'clean': f"{instruction} {clean_expr}",
                'corrupted': f"{instruction} {corrupted_expr}",
                'label': label
            })
            unique_expressions.add(clean_expr) 
            operation_count[operator] += 1

if sum(operation_count.values()) < total_count:
    print(f"only generate {sum(operation_count.values())} ")

df = pd.DataFrame(data)

def format_expression(expr):
    num1, operator, num2 = expr.split()
    return f"{num1} {operator} {num2} = "

df['clean'] = df['clean'].apply(lambda x: f"{instruction.strip()}{format_expression(x.split(':')[-1].strip())}")
df['corrupted'] = df['corrupted'].apply(lambda x: f"{instruction.strip()}{format_expression(x.split(':')[-1].strip())}")

df.to_csv("Logical_Operations_Add_Sub_100.csv", index=False)