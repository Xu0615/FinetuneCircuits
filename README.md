# Finetune Circuits 🥰

**Finetune Circuits** builds on our ICML 2025 paper “Towards Understanding Fine‑Tuning Mechanisms of LLMs via Circuit Analysis,” a **Mechanistic‑Interpretability study** that finds the structural dynamics of Large Language Models under fine‑tuning. We derive interpretability insights that guide targeted modifications—resulting in a **circuit-aware low-rank adaptation ** method that demonstrably improves accuracy over vanilla LoRA.

------

## :dart: Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/Xu0615/FinetuneCircuits.git
   cd FinetuneCircuits
   ```

2. **Create a virtual environment & install dependencies**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate       # Linux / macOS
   .venv\Scripts\activate        # Windows
   
   pip install -r requirements.txt
   ```

------

## :book: Data Preparation & Task Templates

FinetuneCircuits supports multiple tasks; see `2_arithmetic_operations_100/` as a reference template. To add a new task:

1. **Generate datasets** (clean & corrupted) via:

   ```bash
   python create_math_data.py 
   ```

2. **Split data** (80% fine-tune / 20% circuit analysis):

   ```bash
   split_data.ipynb
   ```

Use the scripts in `2_arithmetic_operations_100/` to set up any new task directory (e.g., logical reasoning, custom benchmarks).

------

## 🚩Overview of the Four‑Phase Workflow

1. **Dataset Construction**
    Prepare clean and perturbed datasets for your target tasks, following the naming and split conventions.

2. **Pretrained Model Circuit Extraction**
    Run circuit discovery (EAP‑IG) on a pretrained LLM to identify the base subcircuit graph.

   ```bash
   2_arithmetic_operations_100/find_circuit_100/100_2arithmetic_operations_circuit.ipynb
   ```

3. **Fine‑Tuning & Stage‑Wise Circuit Tracking**

   - Perform parameter‑efficient fine‑tuning (e.g., LoRA) on tasks.  Save each checkpoint during the training process (for example, 10)
   - At checkpoints, re-extract subcircuits to observe dynamic changes through training.

4. **Insights‑Driven CircuitLoRA**

   - Compute edge‑score deltas between pre‑ and post‑fine‑tuning circuits.
   - Retrain with `pythia_topK_lora_simplemath.ipynb`, allocating higher adapter ranks to critical layers for improved performance.

By leveraging insights from circuit dynamics, CircuitLoRA often outperforms vanilla LoRA on accuracy.

------

## 🔍 Circuit Discovery & Evaluation

### 1. Circuit Extraction (EAP‑IG)

```bash
2_arithmetic_operations_100/find_circuit_100/100_2arithmetic_operations_circuit.ipynb
```

Key steps:

- Attribution patching with Integrated Gradients
- Collapse Q/K/V to head‑level nodes via `collapse_qkv.py`
- Export subcircuit graph (edges & nodes)

### 2. Faithfulness & Robustness

- **Faithfulness**: measure subcircuit’s recovery of original outputs (KL‑divergence).
- **Robustness**: perturb inputs (10–90%), re‑extract circuits, compute Jaccard edge similarity.
   Notebook: `stable_analysis/circuit_robust.ipynb`

------

## 🤖 Fine‑Tuning & CircuitLoRA

### Baselines

- **Full‑Parameter Fine‑Tuning** (SFT in `finetune_pythia_*/SFT/`)
- **Vanilla LoRA** (Peft in `finetune_pythia_*/Peft/`)

### CircuitLoRA Workflow

1. **Before Using LoRA** → Extract `C_before`

2. **After Using LoRA** → Extract `C_after`

3. Compute per‑edge score deltas → Aggregate by layer → Select top‑K critical layers (For details, refer to `changecircuit_edge_keylayer.py` and `circuit_weighted_lora.py` in the `Circuit_LoRa` file, which can be directly called in `pythia_topK_lora_simplemath.ipynb`)

4. Retrain with:

   ```bash
   pythia_topK_lora_simplemath.ipynb
   ```

------

## 📈 Graphical Analysis

Visualize training and circuit metrics in `graph_analysis_<task_dir>/`:

- Model accuracy vs. checkpoint
- Circuit faithfulness vs. checkpoint
- Node & edge changes over time

------

## 🔗 Compositional Tasks

For multi-step or compositional reasoning tasks (e.g., `(a op1 b) op2 c`), follow the **add_sub_mul_div** template:

1. **Subtask Circuit Extraction**
   - Extract circuits for each subtasks (e.g., addition and subtraction) using the standard discovery pipeline.
2. **Union Circuit Construction**
   - In `Circuit_Analysis/merge_circuit.ipynb`, merge the extracted subtask circuits into a single **union circuit**.
   - Ensure the merged union circuit has the same number of edges as the direct composite-circuit extracted from the combined task, to enable fair comparison.
3. **CircuitLoRA Evaluation**
   - Fine-tune the composite task twice:
     - Once with the **union circuit** as the circuit prior.
     - Once with the **compositional circuit** (directly extracted composite circuit from compositional reasoning tasks).
   - Compare performance to assess whether the union circuit can effectively replace the expensive composite-circuit discovery step.

By leveraging precomputed subtask circuits and merging them, you can skip resource-intensive circuit extraction on complex composite tasks while still achieving near-equivalent fine-tuning gains.

------

## 🎓 Citation

If you use **FinetuneCircuits** in your research, please cite:

> **Xu Wang et al.** “Towards Understanding Fine‑Tuning Mechanisms of LLMs via Circuit Analysis.” *ICML 2025.*

------

## 🤝 Contributing

**Acknowledgement:** We thank the [EAP-IG repository](https://github.com/hannamw/EAP-IG) for pioneering the circuit discovery methods we build upon.

We welcome issues, PRs, and new task integrations:

- **New tasks:** follow the `2_arithmetic_operations_100/` template.
- **Features & improvements:** fork, develop, and reference relevant notebooks/scripts.

Thank you for your contributions and insights!
