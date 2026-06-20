# Pitch de 10 minutos

## 1. Problema

Vidro quebrado misturado ao lixo e um risco fisico real. A falha humana na separacao pode causar acidentes e contaminar a cadeia de reciclagem.

## 2. Solucao

O AI Smart Waste usa visao computacional para analisar uma imagem do residuo e indicar se ha risco de vidro quebrado.

## 3. Demonstracao

1. Abrir a aplicacao Streamlit.
2. Enviar uma imagem sem vidro quebrado.
3. Mostrar classificacao segura.
4. Enviar uma imagem com cacos de vidro.
5. Mostrar alerta de risco e confianca do modelo.

## 4. Arquitetura

Imagem -> pre-processamento -> data augmentation -> MobileNetV2 -> classificacao -> alerta -> metricas.

## 5. Metricas

Mostrar matriz de confusao, precision, recall e F1-score. Dar destaque ao falso negativo, que e o erro mais perigoso nesse problema.

## 6. Proximos passos

Com imagens anotadas, o sistema pode evoluir para deteccao YOLO, marcando exatamente a regiao do vidro quebrado na imagem.
