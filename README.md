================================================================================
ARTEFATOS DO PROJETO
================================================================================

1. DATA/
--------
Contém os datasets organizados para treinamento, validação e teste.

- data/df/       : Dataset DeepfakeFaces dividido em treino/validacao/teste
- data/foren/    : Dataset Foren dividido em treino/validacao/teste
- data_old/      : Datasets originais antes do split estratificado

Estrutura interna: [dataset]/[split]/[classe]
  - split: treino, validacao, teste
  - classe: fake, real

NOTA: Os datasets devem ser baixados manualmente do Google Drive:
https://drive.google.com/drive/folders/1sGoJ5OSpr4fEnkiIthUnQj39mVRNN88P

2. UTILS/
---------
Scripts utilitários para pré-processamento e compressão de modelos.

- split_df.py : Divide dataset em treino/validação/teste com split estratificado
  Parâmetros:
    --src_root : Pasta origem com faces_224 e metadata.csv (padrão: data_old/df)
    --dst_root : Pasta destino (padrão: data/df)
    --move     : Mover arquivos em vez de copiar (flag)
    --seed     : Seed para reprodutibilidade (padrão: 42)
  
  Execução: python utils/split_df.py --src_root data_old/df --dst_root data/df

- split_foren.py : Divide dataset Foren em treino/validação/teste aplicando novo split estratificado 80/10/10 sobre CADA subset original
  Parâmetros:
    --src_root : Pasta origem com estrutura pessoa (padrão: data_old/foren)
    --dst_root : Pasta destino (padrão: data/foren)
    --move     : Mover arquivos em vez de copiar 
    --seed     : Seed para reprodutibilidade (padrão: 42)
  
  Split aplicado: 80% treino, 10% validação, 10% teste (estratificado por subset)
  Execução: python utils/split_foren.py --src_root data_old/foren --dst_root data/foren

- model_compress.py : Funções para pruning e quantização de modelos
  Funções principais:
    - aplica_pruning() : Aplica L1 pruning não estruturado
      Parâmetros: prune_amount (padrão: 0.2), incluir_convs (padrão: False)
    
    - aplica_quantizacao_estatica() : Quantização estática INT8
      Parâmetros: model_cpu, val_loader_para_calibracao, input_size
      Backend: fbgemm (Linux/Windows) ou qnnpack (macOS)

- augmentation.py : Transformações customizadas para data augmentation
  - RandomJPEGReencode : Aplica recompressão JPEG aleatória
  - RandomCenterCropResize : Crop central com escala aleatória

3. SRC/
-------
Código-fonte para treinamento e teste dos modelos.

Estrutura: src/[modelo]/[versão]/[script].py

MODELOS IMPLEMENTADOS:
- vgg4/        : VGG customizada com 4 blocos convolucionais
- vgg16/       : VGG16 pré-treinada 
- xceptionNet/ : Xception pré-treinada
- mobileNetV3/ : MobileNetV3 Large pré-treinada

VERSÕES:
- V1 : Modelo base sem compressão
- V2 : Modelo com pruning 
- V3 : Modelo com pruning + quantização INT8

SCRIPTS POR VERSÃO:
- treino_foren.py : Treina modelo no dataset Foren
- treino_df.py    : Treina modelo dataset DeepfakeFaces
- teste_foren.py  : Testa modelo no dataset Foren (cross-dataset com DF)
- teste_df.py     : Testa modelo no dataset DF (cross-dataset com Foren)

4. MODELS/
----------
Modelos treinados salvos.

Estrutura: models/[modelo]/[versão]/model_[dataset].pt|pth

Formatos:
- .pt  : PyTorch state_dict (V1, V2)
- .pth : TorchScript quantizado (V3)

Exemplos:
- models/xceptionNet/V1/model_foren.pt  : Xception treinada em Foren
- models/xceptionNet/V3/model_df.pth    : Xception quantizada treinada em DF

NOTA: Os models do vgg16 foram armazenados diretamento no link do Google Drive:
https://drive.google.com/drive/folders/1ZHj-KybbyzjnG6MzsAKFtlYBa0ToWSy4

5. LOGS/
--------
Logs de treinamento e teste.

Estrutura: logs/[modelo]/[versão]/log_[operacao]_[dataset].txt

Exemplos:
- logs/Vgg16/V2/log_treino_foren.txt : Log de treinamento no Foren
- logs/Vgg16/V3/log_teste_df.txt     : Log de teste no DeepfakeFaces (com cross-dataset no foren)

CONTEÚDO:
- GPU utilizada
- Métricas por época (loss, accuracy)
- Classification report (precision, recall, f1-score)
- Tempo de execução

