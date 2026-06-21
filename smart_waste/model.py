"""Define a arquitetura da rede neural usada no projeto.

Aqui temos o modelo de Rede Neural Convolucional (CNN)
Foi usado o MobileNetV2, uma CNN leve e eficiente.
"""

import tensorflow as tf

from smart_waste.config import IMAGE_SIZE


def build_model(num_classes: int) -> tf.keras.Model:
    """Cria e compila o modelo de classificacao de imagens.

    Args:
        num_classes: Quantidade de classes do dataset. Neste projeto,
            normalmente sao 2 classes: broken_glass e safe_waste.

    Returns:
        Um modelo Keras pronto para treinar com model.fit().
    """
    # Camada de entrada: cada imagem chega redimensionada para 224x224 pixels
    # e com 3 canais de cor: vermelho, verde e azul (RGB).
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))

    # Data augmentation para criar pequenas variacoes nas imagens durante o treino.
    augmentation = tf.keras.Sequential(
        [
            # Espelha horizontalmente algumas imagens.
            tf.keras.layers.RandomFlip("horizontal"),
            # Rotaciona levemente as imagens.
            tf.keras.layers.RandomRotation(0.08),
            # Aproxima ou afasta um pouco a imagem.
            tf.keras.layers.RandomZoom(0.12),
            # Altera um pouco o contraste para simular iluminacao diferente.
            tf.keras.layers.RandomContrast(0.15),
        ],
        name="augmentation",
    )

    # Aplica as camadas de aumento de dados.
    x = augmentation(inputs)

    # MobileNetV2 espera uma normalizacao especifica. Esta funcao converte os
    # valores dos pixels para o formato usado quando ela foi pre-treinada.
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

    # MobileNetV2 é uma CNN ja treinada no ImageNet. Usamos a parte
    # convolucional dela como extratora de caracteristicas visuais.
    #
    # include_top=False remove a cabeca original de classificacao do ImageNet.
    # Pois o precisamos identificar o vidro quebrado, não as 1000 classes do ImageNet.
    base_model = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMAGE_SIZE, 3),
    )

    # Congela os pesos da MobileNetV2. Assim, no treino inicial, aprendemos
    # apenas as camadas finais do nosso classificador. Isso reduz custo e risco
    # de overfitting, especialmente com dataset pequeno/medio.
    base_model.trainable = False

    # Passa a imagem pela base convolucional. training=False mantem camadas
    # internas, como BatchNormalization, no comportamento de inferencia.
    x = base_model(x, training=False)

    # Transforma os mapas de caracteristicas em um vetor compacto.
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    # Dropout desliga aleatoriamente 35% dos neuronios durante o treino.
    x = tf.keras.layers.Dropout(0.35)(x)
    # Camada final: gera uma probabilidade para cada classe.
    # softmax garante que as probabilidades somem 1.
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    # Junta entrada e saida em um objeto Model do Keras.
    model = tf.keras.Model(inputs, outputs, name="smart_waste_glass_detector")

    model.compile(
        # Usamos o Adam pois é um otimizador robusto e comum para CNNs.
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        # categorical_crossentropy e usada porque os labels estao em formato
        # one-hot, por exemplo: broken_glass=[1,0], safe_waste=[0,1].
        loss="categorical_crossentropy",
        metrics=[
            # Accuracy mede acertos totais.
            "accuracy",
            # Precision mede, entre os alertas de uma classe, quantos estavam corretos. Ajuda a entender falsos positivos.
            tf.keras.metrics.Precision(name="precision"),
            # Recall mede, entre os casos reais de uma classe, quantos o modelo encontrou.
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model
