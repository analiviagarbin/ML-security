import numpy as np

from matplotlib import pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

from art.utils import load_mnist
from art.attacks.evasion import ZooAttack # importa os algorítimos de ataques
from art.estimators.classification import SklearnClassifier

import warnings
warnings.filterwarnings('ignore')

# carrega os dados do mnist
# <x_imagens-de-treinamento>, <y_label>
(x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist()

x_train.shape
y_train.shape

# o modelo de árvore de decisão aceita o formato 728
# o código ajusta o formato padrão do mnist (28x28) para o aceito pelo modelo
nsamples, nx, ny, nz = x_train.shape
x_train_ok = x_train.reshape((nsamples,nx*ny))

nsamples, nx, ny, nz = x_test.shape
x_test_ok = x_test.reshape((nsamples,nx*ny))

y_train = np.argmax(y_train, axis=1) # np.argmax pega o vetor e pega o índice de maior valor (no caso onde tem 1)
y_test = np.argmax(y_test, axis=1) # originalmente é um vetor de 0 e 1

# criando o modelo (model)
model = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=10,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, random_state=None,
            splitter='best')

# treina o modelo com base de dados do mnist
# <imagens_redimensionadas>, <labels>
model.fit(X=x_train_ok, y=y_train)

# teste de acuracia do modelo
score = model.score(x_test_ok, y_test)
print("Model Score: %.4f" % score)

pred = model.predict(x_test_ok)
acc = accuracy_score(pred, y_test)
print('Test Accuracy : \033[32m \033[01m {:.5f}% \033[30m \033[0m'.format(acc*100))
print(classification_report(y_test, pred))

# faz uma inferencia no modelo
prediction = model.predict(x_test_ok[0].reshape(1,-1))
print("Test Predicted Label: %i" % prediction)

## Cria e exibe um grid com as primeiras 25 (de 10000) imagens e as predições do modelo
import numpy as np
import matplotlib.pyplot as plt

fig, axs = plt.subplots(nrows=5, ncols=5, figsize=(10, 10))

for ax in axs.flat:
    ax.axis('off')

# populando cada subplot
for i in range(5):
    for j in range(5):
        axs[i, j].matshow(x_test_ok[i+j].reshape(28,28))
        prediction = model.predict(x_test_ok[i+j].reshape(1,-1))
        axs[i, j].set_title(f"classified as: {prediction}")

# exibe o grid
plt.tight_layout()
plt.show()


## Atacando o modelo
# O ataque se baseia em causar perturbações mínimas nos pixels das imagens de teste
# sem que a imagem seja alterada para algo completamente diferente
# mas ainda sim o modelo de ML sofra perturbações e devolva inferências incorretas

art_classifier = SklearnClassifier(model=model) # passa o modelo atual para o algoritmo de ataque

zoo = ZooAttack(classifier=art_classifier, confidence=0.0, targeted=False, # False - nao tem uma classe alvo
                learning_rate=1e-1, max_iter=100,
                binary_search_steps=20, initial_const=1e-3, abort_early=True, use_resize=False,
                use_importance=False, nb_parallel=10, batch_size=1, variable_h=0.25) # gera o ataque no modelo "atacável" - art_classifier

# reduz o conjunto para ser mais rápido
x_train_to_attack = x_train_ok[0:200]
x_test_to_attack = x_test_ok[0:200]

# debug
x_test_to_attack.shape
x_train_to_attack.shape

x_train_adv = zoo.generate(x_train_to_attack) # ataque de fato <generate> (gera as amostras para ataque)
# <x_train_adv> imagens atacadas

## Análise de acurácia no modelo atacado
prediction = model.predict(x_train_ok[0:1, :])[0] # pega uma imagem original
print("Adversarial Test Predicted Label: %i" % prediction)

prediction = model.predict(x_train_adv[0:1, :])[0] # pega uma imagem atacada de mesmo índice
print("Adversarial Test Predicted Label: %i" % prediction)

# mostra a imagem atacada
plt.matshow(x_train_adv[0].reshape(28,28))
plt.clim(0, 1)

## Cria um grid com as 25 primeiras imagens atacadas e as inferencias do modelo
import numpy as np
import matplotlib.pyplot as plt


fig, axs = plt.subplots(nrows=5, ncols=5, figsize=(10, 10))

for ax in axs.flat:
    ax.axis('off')

for i in range(5):
    for j in range(5):
        axs[i, j].matshow(x_train_adv[i+j].reshape(28,28))
        prediction = model.predict(x_train_adv[i+j].reshape(1,-1))
        axs[i, j].set_title(f"classified as: {prediction}")

plt.tight_layout()
plt.show()