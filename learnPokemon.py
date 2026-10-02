# Dataset: https://www.kaggle.com/datasets/rounakbanik/pokemon  (pokemon.csv)

import ast
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             ConfusionMatrixDisplay, classification_report)

USAR_AGAINST = False    # True = inclui as colunas against_* (quase entregam o tipo)
N_EPOCAS = 500          # máximo de épocas
PACIENCIA = 100          # para se a validação não melhorar por N épocas
GRAFICO_AO_VIVO = True  # True = atualiza o gráfico durante o treino
ATUALIZA_A_CADA = 10    # de quantas em quantas épocas redesenhar


#%% CARGA DOS DADOS

df = pd.read_csv('pokemon.csv')
y = df['type1']


#%% PRÉ-PROCESSAMENTO

colunas_numericas = [
    'hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed',
    'height_m', 'weight_kg', 'base_total', 'base_egg_steps',
    'base_happiness', 'experience_growth', 'percentage_male',
    'capture_rate', 'generation', 'is_legendary'
]
if USAR_AGAINST:
    colunas_numericas += [c for c in df.columns if c.startswith('against_')]

X_num = df[colunas_numericas].copy()
X_num['capture_rate'] = pd.to_numeric(
    X_num['capture_rate'].astype(str).str.extract(r'(\d+)')[0])
X_num['percentage_male'] = X_num['percentage_male'].fillna(-1)
X_num = X_num.fillna(X_num.median())

X_tipo2 = pd.get_dummies(df['type2'].fillna('none'), prefix='type2', dtype=int)

lista_hab = df['abilities'].apply(ast.literal_eval)
mlb = MultiLabelBinarizer()
X_hab = pd.DataFrame(mlb.fit_transform(lista_hab),
                     columns=['hab_' + h for h in mlb.classes_],
                     index=df.index)

X = pd.concat([X_num, X_tipo2, X_hab], axis=1)
print('Matriz de entrada X:', X.shape)


#%% DIVISÃO: TREINO / VALIDAÇÃO / TESTE

estratificar = y if y.value_counts().min() >= 2 else None
X_tr, X_test, y_tr, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=estratificar)

# validação (10% do treino): serve para escolher a melhor época sem "espiar" o teste
X_train, X_val, y_train, y_val = train_test_split(
    X_tr, y_tr, test_size=0.1, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)

print(f'Treino: {len(X_train)} | Validação: {len(X_val)} | Teste: {len(X_test)}')


#%% TREINAMENTO ÉPOCA POR ÉPOCA

mlp = MLPClassifier(hidden_layer_sizes=(100, 50), alpha=1e-2,
                    activation='relu', random_state=42)
classes = np.unique(y)

hist = {'perda': [], 'treino': [], 'validacao': [], 'teste': []}
melhor_val, melhor_ep, melhor_modelo = -1, 0, None

if GRAFICO_AO_VIVO:
    plt.ion()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))


def desenhar():
    ax1.clear()
    ax1.plot(hist['perda'], color='tab:red')
    ax1.set_title('Perda (erro) no treino')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Perda')

    ax2.clear()
    ax2.plot(hist['treino'], label='treino')
    ax2.plot(hist['validacao'], label='validação')
    ax2.plot(hist['teste'], label='teste')
    ax2.axvline(melhor_ep, color='gray', ls='--', label=f'melhor época ({melhor_ep})')
    ax2.set_title('Curva de aprendizado (acurácia)')
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Acurácia')
    ax2.set_ylim(0, 1.05)
    ax2.legend(loc='lower right')
    fig.tight_layout()
    if GRAFICO_AO_VIVO:
        plt.pause(0.01)


for ep in range(N_EPOCAS):
    mlp.partial_fit(X_train_s, y_train, classes=classes)   # 1 época

    hist['perda'].append(mlp.loss_)
    hist['treino'].append(accuracy_score(y_train, mlp.predict(X_train_s)))
    hist['validacao'].append(accuracy_score(y_val, mlp.predict(X_val_s)))
    hist['teste'].append(accuracy_score(y_test, mlp.predict(X_test_s)))

    # guarda o melhor modelo segundo a VALIDAÇÃO
    if hist['validacao'][-1] > melhor_val:
        melhor_val, melhor_ep = hist['validacao'][-1], ep
        melhor_modelo = copy.deepcopy(mlp)

    if ep % 10 == 0:
        print(f'Época {ep:4d} | perda {mlp.loss_:.4f} | '
              f'treino {hist["treino"][-1]:.3f} | '
              f'val {hist["validacao"][-1]:.3f} | '
              f'teste {hist["teste"][-1]:.3f}')

    if GRAFICO_AO_VIVO and ep % ATUALIZA_A_CADA == 0:
        desenhar()

    if ep - melhor_ep >= PACIENCIA:
        print(f'\nParada antecipada na época {ep} (sem melhora há {PACIENCIA} épocas).')
        break

desenhar()
if GRAFICO_AO_VIVO:
    plt.ioff()
plt.show(block=False)

print(f'\nMelhor época: {melhor_ep} (validação = {melhor_val:.3f})')
mlp = melhor_modelo   # usa o modelo da melhor época daqui para frente


#%% DESEMPENHO SOBRE O CONJUNTO DE TESTE

y_pred = mlp.predict(X_test_s)
print('Acurácia treino:', accuracy_score(y_train, mlp.predict(X_train_s)))
print('Acurácia teste: ', accuracy_score(y_test, y_pred))
print('\nRelatório por classe:')
print(classification_report(y_test, y_pred, zero_division=0))


#%% MATRIZ DE CONFUSÃO

cm = confusion_matrix(y_test, y_pred, labels=mlp.classes_)
fig2, ax = plt.subplots(figsize=(10, 10))
ConfusionMatrixDisplay(cm, display_labels=mlp.classes_).plot(
    ax=ax, xticks_rotation=90, colorbar=False)
ax.set_title(f'Matriz de confusão (época {melhor_ep})')
plt.tight_layout()
plt.show()