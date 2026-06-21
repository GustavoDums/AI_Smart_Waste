"""Inicializacao do pacote smart_waste.

Quando qualquer modulo de smart_waste e importado, este arquivo roda primeiro.
Usamos isso para configurar caches do TensorFlow/Keras e Matplotlib dentro da
pasta do projeto, evitando erros de permissao em pastas do usuario do Windows.
"""

import os
from pathlib import Path


# Pasta raiz para caches gerados por bibliotecas externas.
_CACHE_DIR = Path("artifacts/cache")

# Keras salva pesos baixados, como MobileNetV2, nesta pasta.
_KERAS_HOME = _CACHE_DIR / "keras"

# Matplotlib salva cache de fontes/configuracoes nesta pasta.
_MPLCONFIGDIR = _CACHE_DIR / "matplotlib"

# Cria as pastas antes de apontar as variaveis de ambiente.
_KERAS_HOME.mkdir(parents=True, exist_ok=True)
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)

# setdefault so define a variavel se ela ainda nao existir. Assim, se um
# usuario avancado configurar outro caminho antes, o codigo respeita.
os.environ.setdefault("KERAS_HOME", str(_KERAS_HOME.resolve()))
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR.resolve()))

