import json
import os
from collections import defaultdict

def load_circuit_json(circuit_json_path):
    with open(circuit_json_path, 'r') as f:
        circuit = json.load(f)
    return circuit

def compute_edge_differences(before_circuit, after_circuit):
    before_edges = before_circuit.get('edges', {})
    after_edges = after_circuit.get('edges', {})
    
    edge_diffs = {}
    for edge_name, after_edge in after_edges.items():
        if edge_name in before_edges:
            before_score = before_edges[edge_name].get('score', 0.0)
            after_score = after_edge.get('score', 0.0)
            diff = abs(after_score - before_score)
            edge_diffs[edge_name] = diff
        else:
            after_score = after_edges[edge_name].get('score', 0.0)
            edge_diffs[edge_name] = abs(after_score)
    
    for edge_name, before_edge in before_edges.items():
        if edge_name not in after_edges:
            before_score = before_edge.get('score', 0.0)
            edge_diffs[edge_name] = abs(before_score)
    
    return edge_diffs

def map_edge_to_layer(edge_name, model_prefix='base_model.model.gpt_neox.layers'):
    parts = edge_name.split('->')
    if len(parts) != 2:
        return None
    
    src, dst = parts
    layer_num = None
    
    if src.startswith('a') and '.h' in src:
        layer_num = int(src[1:].split('.h')[0])
    elif src.startswith('m'):
        layer_num = int(src[1:])
    elif src == 'input':
        layer_num = 0
    else:
        layer_num = None
    
    if layer_num is None:
        if dst.startswith('a') and '.h' in dst:
            layer_num = int(dst[1:].split('.h')[0])
        elif dst.startswith('m'):
            layer_num = int(dst[1:])
        elif dst == 'logits':
            layer_num = 23
        else:
            layer_num = None
    
    return layer_num

def aggregate_diffs_per_layer(edge_diffs, model_prefix='base_model.model.gpt_neox.layers'):
    layer_diffs = defaultdict(float)
    for edge_name, diff in edge_diffs.items():
        layer_num = map_edge_to_layer(edge_name, model_prefix)
        if layer_num is not None:
            layer_diffs[layer_num] += diff
    return layer_diffs

def select_critical_layers(layer_diffs, threshold=None, top_k=None):
    diffs = list(layer_diffs.values())
    if not diffs:
        return []
    
    if threshold is None:
        mean_diff = sum(diffs) / len(diffs)
        std_diff = (sum((x - mean_diff) ** 2 for x in diffs) / len(diffs)) ** 0.5
        threshold = mean_diff + std_diff
    
    critical_layers = [layer for layer, diff in layer_diffs.items() if diff >= threshold]
    
    if top_k is not None and top_k > 0:
        sorted_layers = sorted(layer_diffs.items(), key=lambda x: x[1], reverse=True)
        critical_layers = [layer for layer, _ in sorted_layers[:top_k]]
    
    return critical_layers

def map_layers_to_module_names(critical_layer_nums, model_prefix='base_model.model.gpt_neox.layers', include_additional_modules=False):
    critical_modules = []
    for layer_num in critical_layer_nums:
        attention_qkv = f"{model_prefix}.{layer_num}.attention.query_key_value"
        mlp_h_to_4h = f"{model_prefix}.{layer_num}.mlp.dense_h_to_4h"
        critical_modules.extend([attention_qkv, mlp_h_to_4h])
        if include_additional_modules:
            attention_dense = f"{model_prefix}.{layer_num}.attention.dense"
            mlp_4h_to_h = f"{model_prefix}.{layer_num}.mlp.dense_4h_to_h"
            critical_modules.extend([attention_dense, mlp_4h_to_h])
    return critical_modules

def define_critical_layers_via_edges(before_circuit_json_path, after_circuit_json_path, model_prefix='base_model.model.gpt_neox.layers', threshold=None, top_k=None):
    before_circuit = load_circuit_json(before_circuit_json_path)
    after_circuit = load_circuit_json(after_circuit_json_path)
    edge_diffs = compute_edge_differences(before_circuit, after_circuit)
    layer_diffs = aggregate_diffs_per_layer(edge_diffs, model_prefix)
    critical_layer_nums = select_critical_layers(layer_diffs, threshold, top_k)
    critical_modules = map_layers_to_module_names(critical_layer_nums, model_prefix=model_prefix, include_additional_modules=True)
    return critical_modules
