"""Configuracoes centrais do projeto.

Manter esses valores em um arquivo separado evita repetir caminhos e parametros
em varios scripts. Se precisar mudar tamanho da imagem, batch ou nome do
arquivo do modelo, é só alterar aqui.
"""

from pathlib import Path

# Tamanho padrao das imagens que entram na rede neural.
# O MobileNetV2 (modelo usado no projeto) funciona melhor com 224x224.
IMAGE_SIZE = (224, 224)

# Quantas imagens sao processadas por vez durante treino/avaliacao.
BATCH_SIZE = 32

# Semente usada para tornar embaralhamento e splits mais reprodutiveis.
SEED = 42

# Pastas de saida do projeto.
ARTIFACTS_DIR = Path("artifacts")
REPORTS_DIR = Path("reports")

# Arquivos gerados pelo treino.
MODEL_PATH = ARTIFACTS_DIR / "model.keras"
LABELS_PATH = ARTIFACTS_DIR / "labels.json"
HISTORY_PATH = ARTIFACTS_DIR / "history.csv"
