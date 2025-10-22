import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.quantization as quant
import timm
from torchvision.models import vgg16

def optimize_model(model, model_name=None, prune_amount=None, quantize=True, device='cpu', verbose=True):
    """
    Applies pruning + quantization to a given model (VGG from torchvision or Xception from timm).
    
    Args:
        model (nn.Module): The model to optimize.
        model_name (str): 'vgg' or 'xception'. If None, inferred automatically.
        prune_amount (float): Fraction of weights to prune. Defaults depend on model type.
        quantize (bool): Whether to apply dynamic quantization.
        device (str): 'cpu' or 'cuda'.
        verbose (bool): Whether to print progress information.
    """

    # ----------------------------------------
    # Auto-detect model type
    # ----------------------------------------
    model_type = model_name or model.__class__.__name__.lower()
    if "vgg" in model_type:
        model_type = "vgg"
        prune_amount = prune_amount or 0.2
    elif "xception" in model_type:
        model_type = "xception"
        prune_amount = prune_amount or 0.1
    else:
        raise ValueError("Unsupported model type. Only 'vgg' and 'xception' are supported.")

    model.to(device)
    model.eval()

    if verbose:
        print(f"🔍 Detected model type: {model_type.upper()}")
        print(f"🪚 Applying pruning ({prune_amount*100:.0f}% of weights)...")

    # ----------------------------------------
    # Count params before pruning
    # ----------------------------------------
    def count_nonzero_params(model):
        return sum(p.count_nonzero().item() for p in model.parameters())

    params_before = count_nonzero_params(model)

    # ----------------------------------------
    # Apply pruning
    # ----------------------------------------
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            prune.l1_unstructured(module, name="weight", amount=prune_amount)
            prune.remove(module, "weight")

    # ----------------------------------------
    # Count params after pruning
    # ----------------------------------------
    params_after = count_nonzero_params(model)
    if verbose:
        print(f"✅ Pruning done. Nonzero params: {params_after:,} / {params_before:,}")

    # ----------------------------------------
    # Apply quantization (dynamic)
    # ----------------------------------------
    if quantize:
        if verbose:
            print("🔢 Applying dynamic quantization...")
        model = torch.quantization.quantize_dynamic(
            model, {nn.Linear}, dtype=torch.qint8
        )
        if verbose:
            print("✅ Quantization done.")

    if verbose:
        print(f"💾 Model optimized ({model_type.upper()}) successfully!\n")

    return model
