
import io
import base64
from flask import Flask, render_template, jsonify, request
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  


filename = 'models/avocado_pipeline.pkl'
with open(filename, 'rb') as file:
    pipeline = pickle.load(file)

csv_filename = 'models/avocado.csv'  
dataset = pd.read_csv(csv_filename)

numerical_cols = ['Quality1', 'Quality2', 'Quality3', 'Small Bags', 'Large Bags', 'XLarge Bags', 'year']
categorical_cols = ['type', 'region']

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        try:
          data = {col: request.form.get(col) for col in numerical_cols + categorical_cols}

          input_df = pd.DataFrame([data])

          for col in numerical_cols:
              input_df[col] = pd.to_numeric(input_df[col])
          
          prediction = pipeline.predict(input_df)
          prediction = prediction[0] 
        except Exception as e:
          prediction = f"Error: {e}"
          print(e)
    return render_template('index.html', prediction=prediction, numerical_cols=numerical_cols, categorical_cols=categorical_cols)
    
@app.route('/api/dataset', methods=['GET'])
def get_dataset():
    try:
        # Récupérer les paramètres de page et de limite
        page = int(request.args.get('page', 1))  # Page par défaut = 1
        limit = int(request.args.get('limit', 10))  # Limite par défaut = 10

        # Calculer l'index de début et de fin
        start_idx = (page - 1) * limit
        end_idx = start_idx + limit

        # Extraire la tranche des données
        paginated_data = dataset[start_idx:end_idx].to_dict(orient='records')

        return jsonify(paginated_data), 200
    except Exception as e:
        print(f"Error fetching dataset: {e}")
        return jsonify({"error": "Erreur lors de la récupération des données."}), 500


@app.route('/api/visualization', methods=['GET'])
def get_visualization():
    try:
        
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.hist(dataset['Quality1'], bins=20, color='skyblue', edgecolor='black')

        ax.set_title('Distribution de Quality1')
        ax.set_xlabel('Quality1')
        ax.set_ylabel('Fréquence')

        img = io.BytesIO()
        fig.savefig(img, format='png')
        img.seek(0)

        img_base64 = base64.b64encode(img.getvalue()).decode()

        return render_template('visualization.html', img_base64=img_base64)
    except Exception as e:
        print(f"Error generating visualization: {e}")
        return "Erreur lors de la génération du graphique.", 500
    
print(dataset.columns)


if __name__ == '__main__':
    app.run(debug=True)


