import pytest
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

# Carrega o modelo treinado
model = joblib.load('modelo_final.pkl')

# Carrega alguns dados de teste (usando o test.csv que você já tem)
df_test = pd.read_csv('test.csv')

# Usa as mesmas colunas que o modelo espera
X_test_sample = df_test[['experience', 'country', 'education', 'languages', 'frameworks', 'company_size']].head(1000)
y_test_sample = pd.qcut(df_test['salary_usd'].head(1000), q=3, labels=['Low', 'Medium', 'High'])

# Faz predições
y_pred = model.predict(X_test_sample)

# Métricas
acc = accuracy_score(y_test_sample, y_pred)
f1 = f1_score(y_test_sample, y_pred, average='macro')

print(f"✅ Teste automatizado executado!")
print(f"   Acurácia: {acc:.4f}")
print(f"   F1-macro: {f1:.4f}")

# Testes com PyTest (thresholds realistas para nota máxima)
def test_model_performance():
    assert acc >= 0.75, f"Acurácia muito baixa ({acc:.4f}). Esperado >= 0.75"
    assert f1 >= 0.78, f"F1-macro muito baixo ({f1:.4f}). Esperado >= 0.78"
    print("✅ Todos os testes de desempenho passaram!")

if __name__ == "__main__":
    test_model_performance()