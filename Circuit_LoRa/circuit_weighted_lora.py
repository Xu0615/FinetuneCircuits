import torch
import torch.nn as nn
import torch.nn.init as init
import math

class LoRALinear(nn.Module):
    """
    Standard LoRA implementation to replace the original Linear layer.
    """
    def __init__(self, original_linear, r=4, alpha=1.0, dropout=0.0):
        super(LoRALinear, self).__init__()
        self.original_linear = original_linear  # The original Linear layer
        self.r = r
        self.alpha = alpha
        self.scaling = self.alpha / self.r
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        
        # Initialize LoRA parameters
        self.lora_A = nn.Parameter(torch.empty(r, original_linear.in_features))
        self.lora_B = nn.Parameter(torch.empty(original_linear.out_features, r))
        init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))  # Kaiming uniform initialization for lora_A
        init.zeros_(self.lora_B)                            # Initialize lora_B to zero
        
        # Freeze the parameters of the original linear layer
        for param in self.original_linear.parameters():
            param.requires_grad = False

    def forward(self, x):
        # Original linear transformation
        original_output = self.original_linear(x)
        
        # Apply Dropout to the input x
        x_dropped = self.dropout(x)
        
        # LoRA adaptation
        lora_output = x_dropped @ self.lora_A.t() @ self.lora_B.t() * self.scaling
        
        # Combine outputs
        return original_output + lora_output


class EnhancedLoRALinear(nn.Module):
    """
    Enhanced LoRA implementation for critical layers.
    Only uses a larger rank LoRA adaptation module, removing the standard LoRA part.
    """
    def __init__(self, original_linear, extra_r=8, critical_alpha=16.0, dropout=0.0):
        super(EnhancedLoRALinear, self).__init__()
        self.original_linear = original_linear  # The original Linear layer
        self.extra_r = extra_r
        self.critical_alpha = critical_alpha
        self.scaling_extra = self.critical_alpha / self.extra_r
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        
        # Initialize extra LoRA parameters
        self.lora_A_extra = nn.Parameter(torch.empty(self.extra_r, original_linear.in_features))
        self.lora_B_extra = nn.Parameter(torch.empty(original_linear.out_features, self.extra_r))
        init.kaiming_uniform_(self.lora_A_extra, a=math.sqrt(5))
        init.zeros_(self.lora_B_extra)
        
        # Freeze the parameters of the original linear layer
        for param in self.original_linear.parameters():
            param.requires_grad = False

    def forward(self, x):
        # Original linear transformation
        original_output = self.original_linear(x)
        
        # Apply Dropout to the input x
        x_dropped = self.dropout(x)
        
        # Use only the extra LoRA adaptation
        lora_extra_output = x_dropped @ self.lora_A_extra.t() @ self.lora_B_extra.t() * self.scaling_extra
        
        # Combine outputs
        return original_output + lora_extra_output


def _replace_module(model, module_name, new_module):
    """
    Replace a specific module in the model.
    
    Args:
        model (nn.Module): Target model.
        module_name (str): Name of the module to replace (dot-separated path).
        new_module (nn.Module): New module to replace with.
    """
    modules = module_name.split('.')
    parent = model
    for sub_name in modules[:-1]:
        parent = getattr(parent, sub_name)
    setattr(parent, modules[-1], new_module)


def apply_circuit_weighted_lora(model, critical_layers, r=4, alpha=16.0, extra_r=8, critical_alpha=16.0, dropout=0.05):
    """
    Apply LoRA or enhanced LoRA only to critical layers.

    Args:
        model (nn.Module): Model where LoRA will be applied.
        critical_layers (list of str): List of critical layer names.
        r (int): Rank for non-critical layer LoRA.
        alpha (float): Scaling factor for non-critical layer LoRA.
        extra_r (int): Rank for enhanced LoRA for critical layers.
        critical_alpha (float): Scaling factor for critical layer LoRA.
        dropout (float): Dropout probability for LoRA.
    
    Returns:
        nn.Module: Model with LoRA applied.
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if 'embed_out' in name:
                continue
            if name in critical_layers:
                # Replace with enhanced LoRA linear layer using larger r
                enhanced_lora = EnhancedLoRALinear(
                    original_linear=module,
                    extra_r=extra_r,
                    critical_alpha=critical_alpha,
                    dropout=dropout
                )
                _replace_module(model, name, enhanced_lora)
                print(f"Replaced {name} with EnhancedLoRALinear (extra_r={extra_r}, critical_alpha={critical_alpha})")
            else:
                # Replace with standard LoRA linear layer
                lora = LoRALinear(
                    original_linear=module,
                    r=r,
                    alpha=alpha,
                    dropout=dropout
                )
                _replace_module(model, name, lora)
                print(f"Replaced {name} with LoRALinear (r={r}, alpha={alpha})")
    return model


def freeze_non_critical_layers(model, critical_layers):
    """
    Freeze parameters in the model except for LoRA parameters.

    Args:
        model (nn.Module): Model where parameters will be frozen.
        critical_layers (list of str): List of critical layer names.
    
    Returns:
        nn.Module: Model with non-LoRA parameters frozen.
    """
    for name, param in model.named_parameters():
        if 'lora_' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    return model


def circuit_regularization(model, critical_layers, initial_params, lambda_reg=1e-3):
    """
    Apply regularization to critical layer parameters to prevent deviation from their initial values.

    Args:
        model (nn.Module): Model where regularization will be applied.
        critical_layers (list of str): List of critical layer names.
        initial_params (dict): Dictionary of initial parameters for critical layers.
        lambda_reg (float): Regularization strength.
    
    Returns:
        torch.Tensor: Regularization loss.
    """
    reg_loss = 0.0
    for name, param in model.named_parameters():
        if any(name.startswith(critical_layer) for critical_layer in critical_layers):
            initial_param = initial_params.get(name)
            if initial_param is not None:
                initial_param = initial_param.to(param.device)
                reg_loss += torch.nn.functional.mse_loss(param, initial_param)
    return lambda_reg * reg_loss


def save_initial_params(model, critical_layers):
    """
    Save initial parameters of critical layers.

    Args:
        model (nn.Module): Model from which parameters will be saved.
        critical_layers (list of str): List of critical layer names.
    
    Returns:
        dict: Dictionary of initial parameters for critical layers.
    """
    initial_params = {}
    for name, param in model.named_parameters():
        if any(name.startswith(critical_layer) for critical_layer in critical_layers):
            initial_params[name] = param.clone().detach()
    return initial_params