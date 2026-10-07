# PokeLearn

Rede neural (MLP do scikit-learn) que tenta prever o **tipo primário** de um Pokémon a partir dos seus atributos (status, altura, peso, habilidades, tipo secundário etc.).

Dataset: [The Complete Pokemon Dataset (Kaggle)](https://www.kaggle.com/datasets/rounakbanik/pokemon). O arquivo `pokemon.csv` já está no repositório.

## Estrutura

| Arquivo | Descrição |
|---|---|
| `learnPokemon.py` | Versão em script Python |
| `learnPokemon.ipynb` | Mesma lógica em Jupyter Notebook, separada em células |
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

O treino abre uma janela com as curvas de perda e acurácia, atualizada durante as épocas. Ao final, mostra o relatório por classe no terminal e a matriz de confusão.

> Em ambientes sem interface gráfica (servidor, WSL sem WSLg), rode com `MPLBACKEND=Agg python learnPokemon.py`. Nesse caso os gráficos não são exibidos, só a saída no terminal.

## 3b. Rodar pelo Jupyter Notebook

O Jupyter não faz parte do `requirements.txt`. Instale no mesmo `.venv`:

```bash
pip install notebook
jupyter notebook learnPokemon.ipynb
```

Depois, execute as células em ordem (ou **Run → Run All Cells**).

**No VS Code:** instale a extensão *Jupyter*, abra o `learnPokemon.ipynb` e selecione o kernel do `.venv`. Se o VS Code pedir, instale o `ipykernel` (`pip install ipykernel`).

## Configurações

No início do script e na primeira célula do notebook:

| Variável | Padrão | Descrição |
|---|---|---|
| `USAR_AGAINST` | `False` | Inclui as colunas `against_*`, que quase entregam o tipo |
| `N_EPOCAS` | `500` | Número máximo de épocas |
| `PACIENCIA` | `100` | Para o treino se a validação não melhorar por N épocas |
| `GRAFICO_AO_VIVO` | `True` | Atualiza o gráfico durante o treino |
| `ATUALIZA_A_CADA` | `10` | Intervalo de épocas entre as atualizações do gráfico |
