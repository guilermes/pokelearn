# Dataset: https://www.kaggle.com/datasets/rounakbanik/pokemon  (pokemon.csv)
# Alvo (y): type1 -> tipo primário do Pokémon (18 classes)

import ast
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MultiLabelBinarizer
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             ConfusionMatrixDisplay)
from sklearn.exceptions import ConvergenceWarning

USAR_AGAINST = False     # True = inclui as colunas against_* (quase entregam o tipo)
MAX_ITER = 2000          # máximo de épocas (alto, para o treino parar "por falta de melhoria")
N_ITER_NO_CHANGE = 10    # épocas sem melhoria (tol) até parar - padrão do scikit-learn
LR_INIT_SGD = 0.1        # taxa de aprendizado inicial do sgd (com o padrão 0.001 ele quase não aprende)
SEED = 42
PASTA_SAIDA = 'resultados'

ARQUITETURAS = [         # a1) uma camada oculta / a2) duas camadas ocultas
    (10,), (20,), (50,), (100,), (200,),
    (10, 10), (20, 20), (50, 50),
]
ATIVACOES = ['relu', 'logistic']   # 'logistic' é a sigmoid no scikit-learn

EM_NOTEBOOK = 'ipykernel' in sys.modules
os.makedirs(PASTA_SAIDA, exist_ok=True)
pd.set_option('display.width', 250)
pd.set_option('display.max_columns', 30)


#%% CARGA DOS DADOS

df = pd.read_csv('pokemon.csv')
y = df['type1']

print(f'Instâncias (linhas): {df.shape[0]} | Variáveis (colunas): {df.shape[1]}')
print(f'Alvo: type1 com {y.nunique()} classes')
print(y.value_counts().to_string())


#%% PRÉ-PROCESSAMENTO: NUMÉRICOS x NOMINAIS

# --- atributos numéricos -> StandardScaler
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
X_num['percentage_male'] = X_num['percentage_male'].fillna(-1)   # -1 = sem gênero
X_num = X_num.fillna(X_num.median())

scaler = StandardScaler()
X_num = pd.DataFrame(scaler.fit_transform(X_num),
                     columns=colunas_numericas, index=df.index)

# --- atributos nominais -> OneHotEncoder
colunas_nominais = ['type2']
ohe = OneHotEncoder(sparse_output=False)
X_nom = pd.DataFrame(ohe.fit_transform(df[colunas_nominais].fillna('none')),
                     columns=ohe.get_feature_names_out(colunas_nominais),
                     index=df.index)

# 'abilities' é multi-rótulo (lista de habilidades por Pokémon), então o
# OneHotEncoder não se aplica: o MultiLabelBinarizer faz o equivalente
# (uma coluna 0/1 por habilidade)
lista_hab = df['abilities'].apply(ast.literal_eval)
mlb = MultiLabelBinarizer()
X_hab = pd.DataFrame(mlb.fit_transform(lista_hab),
                     columns=['abilities_' + h for h in mlb.classes_],
                     index=df.index)

# --- une numéricos e nominais em uma única matriz X
X = pd.concat([X_num, X_nom, X_hab], axis=1)
print(f'Numéricos: {X_num.shape[1]} | Nominais (one-hot): {X_nom.shape[1] + X_hab.shape[1]}')
print('Matriz de entrada X:', X.shape)


#%% DIVISÃO: 80% TREINO / 20% TESTE

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y)

print(f'Treino: {len(X_train)} | Teste: {len(X_test)}')


#%% FUNÇÕES AUXILIARES

resultados = []


def rodar(item, camadas, ativacao, solver='adam', learning_rate='constant',
          max_iter=MAX_ITER, learning_rate_init=0.001):
    """Treina um MLP, imprime o resultado e mostra a matriz de confusão."""
    n = len(resultados) + 1
    titulo = (f'Rodada {n:02d} [{item}] - camadas={camadas} | ativação={ativacao} | '
              f'solver={solver} | learning_rate={learning_rate} | '
              f'learning_rate_init={learning_rate_init} | max_iter={max_iter}')
    print('\n' + '=' * len(titulo))
    print(titulo)
    print('=' * len(titulo))

    mlp = MLPClassifier(hidden_layer_sizes=camadas, activation=ativacao,
                        solver=solver, learning_rate=learning_rate,
                        learning_rate_init=learning_rate_init,
                        max_iter=max_iter, n_iter_no_change=N_ITER_NO_CHANGE,
                        random_state=SEED)

    inicio = time.time()
    with warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter('always', ConvergenceWarning)
        mlp.fit(X_train, y_train)
    segundos = time.time() - inicio
    atingiu_max = any(issubclass(a.category, ConvergenceWarning) for a in avisos)

    y_pred = mlp.predict(X_test)
    acc_treino = accuracy_score(y_train, mlp.predict(X_train))
    acc_teste = accuracy_score(y_test, y_pred)
    motivo = ('atingiu max_iter (não convergiu)' if atingiu_max
              else f'falta de melhoria ({N_ITER_NO_CHANGE} épocas sem melhorar)')

    print(f'Épocas executadas: {mlp.n_iter_}  -> parou por {motivo}')
    print(f'Perda final: {mlp.loss_:.4f} | tempo: {segundos:.1f}s')
    print(f'Acurácia treino: {acc_treino:.4f}')
    print(f'Acurácia teste:  {acc_teste:.4f}')

    cm = confusion_matrix(y_test, y_pred, labels=mlp.classes_)
    print('Matriz de confusão (linhas = real, colunas = previsto):')
    print(pd.DataFrame(cm, index=mlp.classes_,
                       columns=[c[:4] for c in mlp.classes_]).to_string())

    fig, ax = plt.subplots(figsize=(8, 8))
    ConfusionMatrixDisplay(cm, display_labels=mlp.classes_).plot(
        ax=ax, xticks_rotation=90, colorbar=False)
    ax.set_title(f'Rodada {n:02d}: {camadas} | {ativacao} | {solver} | '
                 f'{learning_rate}\nacurácia teste = {acc_teste:.4f}', fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(PASTA_SAIDA, f'cm_rodada_{n:02d}.png'), dpi=100)
    if EM_NOTEBOOK:
        plt.show()
    else:
        plt.close(fig)

    resultados.append({
        'rodada': n, 'item': item, 'camadas': str(camadas), 'ativacao': ativacao,
        'solver': solver, 'learning_rate': learning_rate,
        'learning_rate_init': learning_rate_init, 'max_iter': max_iter,
        'epocas': mlp.n_iter_, 'atingiu_max_iter': atingiu_max,
        'acc_treino': round(acc_treino, 4), 'acc_teste': round(acc_teste, 4),
    })
    return mlp


#%% a) ARQUITETURA + b) FUNÇÃO DE ATIVAÇÃO (solver adam)
# a1) uma camada oculta: 10, 20, 50, 100, 200 neurônios
# a2) duas camadas ocultas: 10+10, 20+20, 50+50 neurônios
# b) cada arquitetura com 'relu' e 'logistic' (sigmoid)
# c) o número de épocas até parar por falta de melhoria sai em cada rodada

for camadas in ARQUITETURAS:
    for ativacao in ATIVACOES:
        rodar('a/b', camadas, ativacao)


#%% c) NÚMERO DE ÉPOCAS (max_iter)
# Na melhor configuração de a/b, varia max_iter: com poucas épocas o treino é
# interrompido antes de convergir (ConvergenceWarning); a partir de certo ponto
# ele termina sozinho "por falta de melhoria" e aumentar max_iter não muda nada.

ab = pd.DataFrame(resultados)
melhor_ab = ab.loc[ab['acc_teste'].idxmax()]
camadas_c = tuple(int(n) for n in melhor_ab['camadas'].strip('()').split(',') if n.strip())
print(f"\nMelhor configuração de a/b: {camadas_c} | {melhor_ab['ativacao']} "
      f"(acurácia {melhor_ab['acc_teste']:.4f})")

for max_iter in [50, 100, 200, 500, 1000, 2000]:
    rodar('c', camadas_c, melhor_ab['ativacao'], max_iter=max_iter)


#%% d) TAXA DE APRENDIZADO 'adaptive' (solver sgd)
# learning_rate só tem efeito com solver='sgd' (no 'adam' é ignorado).
# 'constant': taxa fixa; para após N_ITER_NO_CHANGE épocas sem melhoria.
# 'adaptive': quando a perda para de cair, divide a taxa por 5 e continua;
#             só para quando a taxa fica menor que 1e-6.
# Compara os dois em todas as arquiteturas, com a melhor ativação de b.

for camadas in ARQUITETURAS:
    for learning_rate in ['constant', 'adaptive']:
        rodar('d', camadas, melhor_ab['ativacao'], solver='sgd',
              learning_rate=learning_rate, learning_rate_init=LR_INIT_SGD)


#%% CONCLUSÃO: TABELAS DE ACURÁCIA E ÉPOCAS

res = pd.DataFrame(resultados)
res.to_csv(os.path.join(PASTA_SAIDA, 'resultados.csv'), index=False)
ordem = [str(c) for c in ARQUITETURAS]

print('\n\n##### a/b) Acurácia no teste: arquitetura x ativação (solver adam) #####')
tab_ab = res[res['item'] == 'a/b'].pivot(index='camadas', columns='ativacao', values='acc_teste')
print(tab_ab.reindex(ordem).to_string(float_format='%.4f'))
media = tab_ab.mean()
print(f"\nMédia por ativação: relu = {media['relu']:.4f} | logistic (sigmoid) = {media['logistic']:.4f}")

print('\n##### c) Épocas até parar (a/b, max_iter=%d) #####' % MAX_ITER)
print(res[res['item'] == 'a/b'].pivot(index='camadas', columns='ativacao', values='epocas')
      .reindex(ordem).to_string())

print('\n##### c) Variação de max_iter na melhor configuração #####')
print(res[res['item'] == 'c'][['max_iter', 'epocas', 'atingiu_max_iter', 'acc_teste']]
      .to_string(index=False, float_format='%.4f'))

print('\n##### d) Acurácia no teste: arquitetura x learning_rate (solver sgd) #####')
tab_d = res[res['item'] == 'd'].pivot(index='camadas', columns='learning_rate', values='acc_teste')
print(tab_d.reindex(ordem).to_string(float_format='%.4f'))
print('\nÉpocas até parar (solver sgd):')
print(res[res['item'] == 'd'].pivot(index='camadas', columns='learning_rate', values='epocas')
      .reindex(ordem).to_string())

melhor = res.loc[res['acc_teste'].idxmax()]
print('\n##### MELHOR RODADA #####')
print(f"Rodada {melhor['rodada']:02d}: camadas={melhor['camadas']} | ativação={melhor['ativacao']} | "
      f"solver={melhor['solver']} | learning_rate={melhor['learning_rate']} | "
      f"épocas={melhor['epocas']} | acurácia teste={melhor['acc_teste']:.4f}")
print(f"Matriz de confusão: {PASTA_SAIDA}/cm_rodada_{melhor['rodada']:02d}.png")
