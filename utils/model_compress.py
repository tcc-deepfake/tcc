import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.quantization import quantize_dynamic

# ---------- função auxiliar (pruning) ----------
def encontra_podas(model, incluir_convs: bool = False):
    camadas = []
    for m in model.modules():
        if isinstance(m, nn.Linear):
            camadas.append((m, 'weight'))
        elif incluir_convs and isinstance(m, nn.Conv2d):
            camadas.append((m, 'weight'))
    return camadas

# ---------- pruning ----------
def aplica_pruning(model, prune_amount=0.2, incluir_convs=False, verbose=True):
    model.eval()
    alvos = encontra_podas(model, incluir_convs=incluir_convs)
    if not alvos:
        if verbose: print("Nada para podar.")
        return model
    prune.global_unstructured(alvos, pruning_method=prune.L1Unstructured, amount=prune_amount)
    if verbose:
        print(f"--- Pruning {prune_amount*100:.0f}%")
    return model

# ---------- remove pesos originais ----------
def limpa_pesos(model):
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)) and prune.is_pruned(m):
            prune.remove(m, 'weight')
    return model

# ---------- quantização ----------
def aplica_quantizacao(model_pruned, verbose=True):
    if verbose:
        print("--- Aplicando Quantização Dinâmica ---")
        
    model_cpu = model_pruned.to("cpu")
    model_cpu.eval()
    
    return quantize_dynamic(model_cpu, {nn.Linear}, dtype=torch.qint8)

# ---------- checa esparsidade ----------
def check_sparsity(model, verbose=True):
    total_zeros = 0
    total_elements = 0
    
    for name, param in model.named_parameters():
        if 'weight' in name:
            total_zeros += torch.sum(param == 0).item()
            total_elements += param.nelement()
    
    if total_elements > 0:
        sparsity = (total_zeros / total_elements) * 100
        if verbose:
            print(f"Total de Zeros nos Pesos: {total_zeros}")
            print(f"Total de Elementos nos Pesos: {total_elements}")
            print(f"Esparsidade Global: {sparsity:.2f}%")
        print("="*40 + "\n")
        return sparsity
    else:
        if verbose:
            print("Nenhum peso encontrado para verificar.")
        print("="*40 + "\n")
        return 0.0
