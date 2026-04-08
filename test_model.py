import pytest
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

# Carrega o modelo
model = joblib.load('modelo_final.pkl')

def test_model_performance():
    """Teste automatizado que verifica o desempenho do modelo."""
    
    # Carrega dados de teste
    df_test = pd.read_csv('test.csv')
    
    # Seleciona as mesmas colunas usadas no treino
    X_test_sample = df_test[['experience', 'country', 'education', 'languages', 'frameworks', 'company_size']].head(1000)
    y_test_sample = pd.qcut(df_test['salary_usd'].head(1000), q=3, labels=['Low', 'Medium', 'High'])
    
    # Faz predições
    y_pred = model.predict(X_test_sample)
    
    # Calcula métricas
    acc = accuracy_score(y_test_sample, y_pred)
    f1 = f1_score(y_test_sample, y_pred, average='macro')
    
    print(f"✅ Teste executado - Acurácia: {acc:.4f} | F1-macro: {f1:.4f}")
    
    # Thresholds definidos por nós
    assert acc >= 0.75, f"Acurácia abaixo do mínimo esperado: {acc:.4f}"
    assert f1 >= 0.78, f"F1-macro abaixo do mínimo esperado: {f1:.4f}"
    
    print("✅ Todos os testes de desempenho passaram!")

if __name__ == "__main__":
    test_model_performance()