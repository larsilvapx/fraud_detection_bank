# 🛡️ Fraud Detection in Banking Transactions

Sistema de detecção de fraudes em transações bancárias utilizando Machine Learning com Python.

Este projeto foi desenvolvido com foco em boas práticas de Engenharia de Software aplicadas à Ciência de Dados, utilizando uma arquitetura modular, pipeline de processamento e comparação de modelos de Machine Learning.

---

# Objetivos

- Analisar transações bancárias
- Identificar fraudes
- Comparar algoritmos de Machine Learning
- Avaliar métricas de desempenho
- Gerar visualizações automáticas
- Organizar o projeto em uma arquitetura escalável

---

# Tecnologias Utilizadas

- Python 3.11
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Joblib
- Jupyter Notebook
- Git
- GitHub

---

# Estrutura do Projeto

```text
fraud_detection_bank/

├── data/
├── images/
├── models/
├── notebooks/
├── src/
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Pipeline

O projeto segue o seguinte fluxo:

```text
Dataset
      │
      ▼
Carregamento
      │
      ▼
Pré-processamento
      │
      ▼
Treinamento
      │
      ▼
Avaliação
      │
      ▼
Visualizações
      │
      ▼
Relatório
```

---

# Modelos Implementados

- Logistic Regression
- Random Forest

Em breve:

- XGBoost
- LightGBM
- CatBoost

---

# Métricas Avaliadas

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

---

# Estrutura dos Módulos

| Arquivo | Responsabilidade |
|----------|------------------|
| config.py | Configuração do projeto |
| data_loader.py | Carregamento dos dados |
| preprocessing.py | Pré-processamento |
| models.py | Treinamento dos modelos |
| metrics.py | Avaliação |
| visualization.py | Geração dos gráficos |
| pipeline.py | Execução do pipeline |

---

# Como executar

## Clone o projeto

```bash
git clone https://github.com/larsilvapx/fraud_detection_bank
```

## Crie o ambiente virtual

```bash
python -m venv .venv
```

## Ative o ambiente

Windows

```bash
.venv\Scripts\activate
```

Linux/Mac

```bash
source .venv/bin/activate
```

## Instale as dependências

```bash
pip install -r requirements.txt
```

## Execute os notebooks

Abra a pasta notebooks e execute na sequência:

- 01_analise_exploratoria.ipynb
- 02_visualizacoes.ipynb
- 03_resultados.ipynb

---

# Resultados

As métricas dos modelos são comparadas automaticamente.

As imagens geradas são salvas em:

```
images/
```

---

# Próximas melhorias

- Implementação de SMOTE
- XGBoost
- Feature Engineering
- Dashboard em Streamlit
- API com FastAPI
- Deploy em Docker

---

# Autor

**Luciano Arruda Rodrigues da Silva**

Desenvolvedor Python | Java Backend | Data Analytics

LinkedIn:
https://linkedin.com/in/lucasards

GitHub:
https://github.com/SEU-USUARIO
