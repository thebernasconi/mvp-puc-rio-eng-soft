# MVP - Pós-Graduação em Engenharia de Software (PUC-Rio)

## Previsão de Faixa Salarial de Desenvolvedores de Software

**Dataset:** Software Developer Salary Prediction (Kaggle)  
**Problema:** Classificação multiclasse (Low / Medium / High)  
**Melhor modelo:** SVM (F1-macro = 0.7986)

### O que foi entregue

- **Notebook completo** (`1_notebook_ml_model.ipynb`)  
  - Carga via URL  
  - Pipelines + pré-processamento  
  - KNN, Árvore, Naive Bayes e SVM  
  - GridSearchCV + cross-validation  
  - Análise de resultados e reflexão de segurança

- **Aplicação Full Stack** (Flask + HTML/JS)  
  - Modelo embarcado no back-end  
  - Formulário simples no front-end  
  - Predição em tempo real

- **Teste automatizado** (`test_model.py`) com PyTest  
  - Thresholds: acurácia ≥ 0.75 e F1-macro ≥ 0.78

**Vídeo de demonstração:** [insira o link do seu vídeo aqui]

**Como executar localmente:**
```bash
pip install -r requirements.txt
python app.py

Acesse: http://127.0.0.1:5000
Repositório: https://github.com/thebernasconi/mvp-puc-rio-eng-soft