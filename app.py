from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Carrega o modelo treinado (embarcado)
model = joblib.load('modelo_final.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Recebe os dados do formulário
        data = {
            'experience': float(request.form['experience']),
            'country': request.form['country'],
            'education': request.form['education'],
            'languages': request.form['languages'],
            'frameworks': request.form['frameworks'],
            'company_size': request.form['company_size']
        }
        
        # Converte para DataFrame (formato que o modelo espera)
        input_df = pd.DataFrame([data])
        
        # Faz a predição
        prediction = model.predict(input_df)[0]
        
        return jsonify({
            'success': True,
            'prediction': prediction,
            'message': f'A faixa salarial prevista é: <strong>{prediction}</strong>'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Erro: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True)