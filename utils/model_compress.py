import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.ao.quantization
import sys
from torch.ao.quantization import get_default_qconfig
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx

# ---------- função auxiliar (pruning) ----------
def encontra_podas(model, incluir_convs: bool = False):
    model_name = model.__class__.__name__
    
    if model_name == 'VGG' and incluir_convs:
        camadas = []
        VGG_BLOCK_5_START_INDEX = 24
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                camadas.append((module, 'weight'))
            elif isinstance(module, nn.Conv2d):
                if name.startswith('features.'):
                    try:
                        layer_index = int(name.split('.')[-1])
                        if layer_index >= VGG_BLOCK_5_START_INDEX:
                            camadas.append((module, 'weight'))
                    except (ValueError, IndexError):
                        pass
        return camadas

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

# ---------- quantização -----------
def aplica_quantizacao_estatica(model_cpu, val_loader_para_calibracao, input_size=(1, 3, 299, 299), verbose=True):
    model_cpu.eval()
        
    backend = "fbgemm"

    if sys.platform == 'darwin':
        backend = "qnnpack"
        torch.backends.quantized.engine = 'qnnpack'
    else:
        torch.backends.quantized.engine = 'fbgemm'
        
    if verbose:
        print(f"Usando backend de quantização da CPU: {torch.backends.quantized.engine}")

    # 1. Configuração
    qconfig = get_default_qconfig(backend) # Usa o backend correto
    qconfig_mapping = torch.ao.quantization.QConfigMapping().set_global(qconfig)
    example_inputs = (torch.randn(input_size),)

    model_prepared = prepare_fx(model_cpu, qconfig_mapping, example_inputs)

    num_batches_calibracao = 50 
    with torch.no_grad():
        for i, (images, _) in enumerate(val_loader_para_calibracao):
            if i >= num_batches_calibracao:
                break
            model_prepared(images) # Coleta estatísticas
            if verbose and (i+1) % 10 == 0:
                print(f"  Batch de calibração {i+1}/{num_batches_calibracao}")

    model_quantized = convert_fx(model_prepared)
    if verbose:
        print("Modelo convertido para int8")
        
    return model_quantized

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
