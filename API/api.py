from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import logging
from sksurv.linear_model import CoxPHSurvivalAnalysis

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HeartFailurePredictor:
    def __init__(self, data_path: str, model_path: str):
        """Inicializar el predictor con las rutas de datos y modelo."""
        self.features = ['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes',
                        'ejection_fraction', 'high_blood_pressure', 'platelets',
                        'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time']
        
        try:
            self.df = self._load_data(data_path)
            self.model = self._load_model(model_path)
        except Exception as e:
            logger.error(f"Error durante la inicialización: {str(e)}")
            raise

    def _load_data(self, path: str) -> pd.DataFrame:
        """Cargar y validar los datos."""
        try:
            df = pd.read_csv(path)
            df['id'] = df.index
            missing_cols = set(self.features) - set(df.columns)
            if missing_cols:
                raise ValueError(f"Columnas faltantes en el dataset: {missing_cols}")
            return df
        except Exception as e:
            logger.error(f"Error al cargar datos: {str(e)}")
            raise

    def _load_model(self, path: str) -> CoxPHSurvivalAnalysis:
        """Cargar el modelo entrenado."""
        try:
            with open(path, 'rb') as file:
                return pickle.load(file)
        except Exception as e:
            logger.error(f"Error al cargar modelo: {str(e)}")
            raise

    def prepare_data_for_prediction(self, patient_data: dict) -> pd.DataFrame:
        """Preparar datos para predicción."""
        return pd.DataFrame([patient_data])[self.features]

    def predict_survival(self, patient_data: dict) -> dict:
        """Predecir curva de supervivencia para un paciente."""
        X_patient = self.prepare_data_for_prediction(patient_data)
        risk_score = self.model.predict(X_patient)[0]
        
        times = np.linspace(0, 365, 100)
        base_survival = np.exp(-0.1 * times)
        patient_survival = np.power(base_survival, np.exp(risk_score))
        
        return {
            'times': times.tolist(),
            'patient_survival': patient_survival.tolist(),
            'base_survival': base_survival.tolist()
        }

    def get_risk_factors(self, patient_data: dict) -> dict:
        """Calcular factores de riesgo para un paciente."""
        return {
            'Edad': 1 if patient_data['age'] > 60 else 0,
            'Ejección': 1 if patient_data['ejection_fraction'] < 30 else 0,
            'Creatinina': 1 if patient_data['serum_creatinine'] > 1.5 else 0,
            'Presión': 1 if patient_data['high_blood_pressure'] == 1 else 0,
            'Diabetes': 1 if patient_data['diabetes'] == 1 else 0
        }

    def get_patient_data(self, patient_id: int) -> dict:
        """Obtener datos de un paciente específico."""
        try:
            patient = self.df.iloc[patient_id]
            return patient[self.features].to_dict()
        except Exception as e:
            logger.error(f"Error al obtener datos del paciente: {str(e)}")
            raise

    def get_population_stats(self) -> dict:
        """Obtener estadísticas de la población."""
        binary_vars = ['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking']
        return {
            'binary_stats': self.df[binary_vars].mean().to_dict(),
            'numerical_stats': {
                col: {
                    'mean': float(self.df[col].mean()),
                    'std': float(self.df[col].std()),
                    'min': float(self.df[col].min()),
                    'max': float(self.df[col].max())
                } for col in self.features if col not in binary_vars + ['time']
            }
        }

predictor = HeartFailurePredictor(
    data_path="https://raw.githubusercontent.com/jetabaresj/DespliegueTeam23/refs/heads/main/data/heart_failure_clinical_records_dataset.csv",
    model_path="model.pkl")

# Inicializar Flask
app = Flask(__name__)

@app.route('/api/patients/<int:patient_id>', methods=['GET'])
def get_patient(patient_id):
    try:
        patient_data = predictor.get_patient_data(patient_id)
        return jsonify(patient_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/predict/<int:patient_id>', methods=['GET'])
def predict(patient_id):
    try:
        patient_data = predictor.get_patient_data(patient_id)
        survival_data = predictor.predict_survival(patient_data)
        risk_factors = predictor.get_risk_factors(patient_data)
        return jsonify({
            'survival_data': survival_data,
            'risk_factors': risk_factors
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/population-stats', methods=['GET'])
def get_population_stats():
    try:
        stats = predictor.get_population_stats()
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/patients', methods=['GET'])
def get_patients():
    try:
        return jsonify({
            'patient_ids': list(range(len(predictor.df)))
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400
    
@app.route("/api/mortality-correlation", methods=["GET"])
def get_mortality_correlation():
    return jsonify({
        "mortality_rates": {
            'smoking': 0.3125, 
            'anaemia': 0.356, 
            'high_blood_pressure': 0.371, 
            'diabetes':0.32, 
            'sex': 0.319,
            'base_rate': 0.32}
    })

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=8000)
