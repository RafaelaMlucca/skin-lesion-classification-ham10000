# Classificação de Lesões de Pele — HAM10000

Estudo comparativo de arquiteturas de visão computacional para a classificação de lesões de pele
do conjunto **HAM10000** (10.015 imagens dermatoscópicas, 7 classes).

O objetivo não foi encontrar um único melhor modelo, mas obter uma visão geral de como diferentes
famílias de algoritmos se comportam sob o **mesmo protocolo** (mesma divisão dos dados, mesmo
pré-processamento, mesmas métricas). São comparadas duas abordagens:

- **Modelo A** — DINOv2 (ViT-B/14) **congelado** como extrator de características + **SVM** (RBF).
- **Modelo B** — **ajuste fino** de cinco backbones modernos: ConvNeXt V2, EfficientNetV2-S,
  ConvNeXt, Swin Transformer e MaxViT.

---

## Linguagem

Python 3, em notebooks Jupyter executados no **Google Colab** (com GPU).

## Bibliotecas necessárias

- `torch`, `torchvision`
- `timm` (modelos pré-treinados)
- `scikit-learn`
- `kagglehub` (download da base)
- `grad-cam` (mapas Grad-CAM)
- `numpy`, `pandas`, `matplotlib`, `Pillow`

No Colab, a maioria já vem instalada. Basta instalar:

```bash
pip install timm kagglehub grad-cam
```

## Base de dados

**HAM10000** — Kaggle: `kmader/skin-cancer-mnist-ham10000`.
O download é feito automaticamente pelos notebooks via `kagglehub` (não é preciso baixar à mão):

```python
import kagglehub
DATA_DIR = kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000")
```

Observações:
- Em alguns ambientes o `kagglehub` pode pedir credenciais do Kaggle; no Colab costuma funcionar
  direto. Alternativamente, é possível montar a base pelo Google Drive (ajustando `DATA_DIR`).
- O aviso de `HF_TOKEN` ausente do Hugging Face é **inofensivo** — os modelos do `timm` usados são
  públicos. Para silenciá-lo, basta criar um token do tipo *Read* e adicioná-lo como secret no Colab.

## Como executar

1. Abra os notebooks no **Google Colab** e ative a GPU: *Runtime → Change runtime type → GPU (T4)*.
2. Execute os notebooks na ordem abaixo. Cada um instala as dependências e baixa a base nas
   primeiras células.

| Ordem | Notebook | O que faz |
|-------|----------|-----------|
| 1 | `01_eda.ipynb` | Análise exploratória: distribuição das classes, duplicidade por `lesion_id`, idade e amostras de imagens. |
| 2 | `02_experiments.ipynb` | Versão inicial dos experimentos (DINOv2 + SVM e um backbone de referência). |
| 3 | `02_experiments_v2.ipynb` | Versão aprimorada; produz a linha "pesos de classe" da ablação do MixUp. |
| 4 | `03_experiments.ipynb` | Sweep dos cinco backbones com MixUp — **resultados principais** do trabalho. |
| 5 | `04_visualizacoes.ipynb` | t-SNE das características, Grad-CAM, galeria de erros e salvamento de checkpoints. |

Notas práticas:
- O `03` treina vários backbones em sequência e pode esgotar o limite de GPU do Colab. Se isso
  ocorrer, treine menos backbones por sessão (reduza a lista `BACKBONES`); o dicionário `runs`
  acumula os resultados entre execuções.
- O `04` salva checkpoints (`.pt`) e usa uma lógica de *treina-ou-carrega*: aponte `CKPT_DIR` para o
  Google Drive para reaproveitar os pesos sem retreinar.

## Estrutura dos arquivos

```
.
├── README.md
├── 01_eda.ipynb
├── 02_experiments.ipynb
├── 02_experiments_v2.ipynb
├── 03_experiments.ipynb
├── 04_visualizacoes.ipynb
├── relatorio.pdf            # versão compilada
└── figuras/                 # figuras geradas (distribuição, amostras, t-SNE, matriz, erros...)
```

## Reprodutibilidade

- Semente fixa (`SEED = 42`).
- Divisão **agrupada por `lesion_id`** (estratificada), evitando vazamento de imagens da mesma
  lesão entre treino, validação e teste.
- Conjunto de teste com 2.014 imagens; métricas principais: balanced accuracy, macro-F1 e
  revocação por classe (com destaque para a revocação do melanoma).
