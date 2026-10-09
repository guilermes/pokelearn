# PokeLearn

Rede neural (MLP do scikit-learn) que tenta prever o **tipo primário** de um Pokémon a partir dos seus atributos (status, altura, peso, habilidades, tipo secundário etc.).

Dataset: [The Complete Pokemon Dataset (Kaggle)](https://www.kaggle.com/datasets/rounakbanik/pokemon). O arquivo `pokemon.csv` já está no repositório.

- **Base:** 801 instâncias e 41 variáveis
- **Alvo (y):** `type1`, com 18 classes
- **Atributos numéricos** (16): escalados com `StandardScaler`
- **Atributos nominais:** `type2` vira colunas com `OneHotEncoder`. `abilities` tem várias habilidades por Pokémon, então usa o `MultiLabelBinarizer`, que é o equivalente para listas.
- **Divisão:** 80% treino e 20% teste, com `train_test_split` estratificado

## Experimentos (atividade)

O script roda **38 rodadas** de `MLPClassifier`. Cada rodada imprime um título com os parâmetros, as épocas executadas e o motivo da parada, a acurácia e a matriz de confusão.

| Item | O que varia |
|---|---|
| a) Arquitetura | uma camada oculta: 10, 20, 50, 100 e 200 neurônios; duas camadas: 10+10, 20+20 e 50+50 |
| b) Ativação | `relu` x `logistic` (a sigmoid no scikit-learn) em todas as arquiteturas, com solver `adam` |
| c) Épocas | `max_iter` de 50 a 2000 na melhor configuração de a/b. Mostra a partir de quando o treino para sozinho "por falta de melhoria" (10 épocas sem melhorar) |
| d) Taxa de aprendizado | `constant` x `adaptive` em todas as arquiteturas. Usa solver `sgd`, porque o `learning_rate` não tem efeito no `adam` |
| e) Matriz de confusão | uma por rodada |

No fim aparecem tabelas de acurácia (4 casas decimais) e de épocas, além da melhor rodada.

As saídas ficam em `resultados/`:
- `cm_rodada_XX.png`: matriz de confusão de cada rodada
- `resultados.csv`: parâmetros, épocas e acurácias de todas as rodadas

## Estrutura

| Arquivo | Descrição |
|---|---|
| `learnPokemon.py` | Versão em script Python |
| `learnPokemon.ipynb` | Mesma lógica em Jupyter Notebook, separada em células |
| `resultados/` | Matrizes de confusão (PNG) e `resultados.csv` gerados pela última execução |
| `atividade.txt` | Enunciado da atividade |
| `pokemon.csv` | Dataset |
| `requirements.txt` | Dependências do projeto |

## Pré-requisitos

- Python 3.11 ou superior (testado com 3.12)

## 1. Criar e ativar o ambiente virtual (.venv)

Na pasta do projeto:

**Linux / macOS / WSL**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell)**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Windows (cmd)**
```bat
python -m venv .venv
.venv\Scripts\activate.bat
```

Com o ambiente ativo, o prompt mostra `(.venv)`. Para sair, use `deactivate`.

## 2. Instalar as dependências

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 3a. Rodar pelo script Python

```bash
python learnPokemon.py
```

A execução completa leva cerca de 5 minutos. O script não abre janelas: as matrizes de confusão são impressas no terminal e salvas em `resultados/`. Para guardar a saída do terminal para o relatório:

```bash
python learnPokemon.py | tee resultados/saida_terminal.txt
```

## 3b. Rodar pelo Jupyter Notebook

O Jupyter não faz parte do `requirements.txt`. Instale no mesmo `.venv`:

```bash
pip install notebook
jupyter notebook learnPokemon.ipynb
```

Depois, execute as células em ordem (ou **Run → Run All Cells**). Leva cerca de 8 minutos. No notebook, cada matriz de confusão também aparece logo abaixo da rodada.

**No VS Code:** instale a extensão *Jupyter*, abra o `learnPokemon.ipynb` e selecione o kernel do `.venv`. Se o VS Code pedir, instale o `ipykernel` (`pip install ipykernel`).

## Configurações

No início do script e na primeira célula do notebook:

| Variável | Padrão | Descrição |
|---|---|---|
| `USAR_AGAINST` | `False` | Inclui as colunas `against_*`, que quase entregam o tipo |
| `MAX_ITER` | `2000` | Máximo de épocas nos itens a, b e d (alto, para o treino parar por falta de melhoria) |
| `N_ITER_NO_CHANGE` | `10` | Épocas sem melhoria até o treino parar |
| `LR_INIT_SGD` | `0.1` | Taxa de aprendizado inicial do `sgd` no item d. Com o padrão (0.001) a rede quase não aprende |
| `ARQUITETURAS` | 8 arquiteturas | Camadas ocultas testadas |
| `ATIVACOES` | `relu`, `logistic` | Funções de ativação testadas |
| `SEED` | `42` | Semente, para resultados reprodutíveis |
