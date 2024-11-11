from dash import Dash, html, dcc, callback, Output, Input, State
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import logging
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.preprocessing import OneHotEncoder
from sksurv.metrics import concordance_index_censored

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
            # Validar columnas requeridas
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

    def prepare_data_for_prediction(self, patient_data: pd.Series) -> pd.DataFrame:
        """Preparar datos para predicción manteniendo los nombres de las features."""
        # Crear un DataFrame con una sola fila y los nombres de características correctos
        X = pd.DataFrame([patient_data[self.features].to_dict()])
        return X[self.features]  # Asegurar el orden correcto de las columnas

    def predict_survival(self, patient_data: pd.Series) -> tuple:
        """Predecir curva de supervivencia para un paciente."""
        X_patient = self.prepare_data_for_prediction(patient_data)
        risk_score = self.model.predict(X_patient)[0]
        
        times = np.linspace(0, 365, 100)
        base_survival = np.exp(-0.1 * times)
        patient_survival = np.power(base_survival, np.exp(risk_score))
        
        return times, patient_survival, base_survival

    def get_risk_factors(self, patient_data: pd.Series) -> dict:
        """Calcular factores de riesgo para un paciente."""
        return {
            'Edad': 1 if patient_data['age'] > 60 else 0,
            'Ejección': 1 if patient_data['ejection_fraction'] < 30 else 0,
            'Creatinina': 1 if patient_data['serum_creatinine'] > 1.5 else 0,
            'Presión': 1 if patient_data['high_blood_pressure'] == 1 else 0,
            'Diabetes': 1 if patient_data['diabetes'] == 1 else 0
        }

class DashboardUI:
    def __init__(self, predictor: HeartFailurePredictor):
        """Inicializar la interfaz del dashboard."""
        self.predictor = predictor
        self.app = Dash(__name__)
        self._setup_layout()
        self._setup_callbacks()

    def _setup_layout(self):
        """Configurar el layout del dashboard."""
        self.app.layout = html.Div([
            html.H1('Valorador del riesgo de insuficiencia cardiaca',
                   className='dashboard-title'),
            
            # Grid principal
            html.Div([
                # Primera fila
                self._create_first_row(),
                # Segunda fila
                self._create_second_row(),
                # Tercera fila
                self._create_third_row()
            ], className='dashboard-container')
        ])

    def _create_first_row(self):
        """Crear primera fila del dashboard."""
        return html.Div([
            # Panel de información del paciente
            html.Div([
                html.H3('Información del paciente'),
                dcc.Dropdown(
                    id='patient-id',
                    options=[{'label': f'Paciente {i}', 'value': i} 
                            for i in self.predictor.df.index],
                    value=0,
                    className='input-field'
                ),
                html.Div(id='patient-info', className='info-panel')
            ], className='panel'),
            
            # Panel de variables numéricas
            html.Div([
                html.H3('Caracterización de variables numéricas'),
                dcc.Dropdown(
                    id='variable-selector',
                    options=[
                        {'label': 'Edad', 'value': 'age'},
                        {'label': 'Creatinina Fosfoquinasa', 'value': 'creatinine_phosphokinase'},
                        {'label': 'Fracción de Eyección', 'value': 'ejection_fraction'},
                        {'label': 'Plaquetas', 'value': 'platelets'},
                        {'label': 'Creatinina Sérica', 'value': 'serum_creatinine'},
                        {'label': 'Sodio Sérico', 'value': 'serum_sodium'}
                    ],
                    value='age',
                    className='input-field'
                ),
                dcc.Graph(id='distribution-plot')
            ], className='panel')
        ], className='row')

    def _create_second_row(self):
        """Crear segunda fila del dashboard."""
        return html.Div([
            # Panel de características binarias
            html.Div([
                html.H3('Características del paciente vs población'),
                dcc.Graph(id='binary-characteristics')
            ], className='panel'),
            
            # Panel de análisis de riesgo
            html.Div([
                html.H3('Análisis de riesgo'),
                dcc.Graph(id='risk-analysis')
            ], className='panel')
        ], className='row')

    def _create_third_row(self):
        """Crear tercera fila del dashboard."""
        return html.Div([
            html.Div([
                html.H3('Análisis de Supervivencia'),
                dcc.Graph(id='survival-curve')
            ], className='panel full-width')
        ], className='row')

    def _setup_callbacks(self):
        """Configurar todos los callbacks del dashboard."""
        @self.app.callback(
            Output('patient-info', 'children'),
            Input('patient-id', 'value')
        )
        def update_patient_info(patient_id):
            try:
                patient = self.predictor.df.iloc[patient_id]
                return html.Div([
                    html.P(f"Edad: {patient['age']} años"),
                    html.P(f"Sexo: {'Masculino' if patient['sex'] == 1 else 'Femenino'}"),
                    html.P(f"Anemia: {'Sí' if patient['anaemia'] == 1 else 'No'}"),
                    html.P(f"Diabetes: {'Sí' if patient['diabetes'] == 1 else 'No'}"),
                    html.P(f"Presión Alta: {'Sí' if patient['high_blood_pressure'] == 1 else 'No'}"),
                    html.P(f"Fumador: {'Sí' if patient['smoking'] == 1 else 'No'}")
                ])
            except Exception as e:
                logger.error(f"Error en update_patient_info: {str(e)}")
                return html.Div("Error al cargar información del paciente")

        @self.app.callback(
            Output('distribution-plot', 'figure'),
            [Input('variable-selector', 'value'),
             Input('patient-id', 'value')]
        )
        def update_distribution(selected_var, patient_id):
            try:
                patient_value = self.predictor.df.iloc[patient_id][selected_var]
                
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=self.predictor.df[selected_var],
                    name='Población',
                    nbinsx=30,
                    histnorm='probability density',
                    opacity=0.7
                ))
                
                fig.add_vline(
                    x=patient_value,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Paciente: {patient_value:.1f}",
                    annotation_position="top right"
                )
                
                fig.update_layout(
                    title=f'Distribución de {selected_var}',
                    xaxis_title=selected_var,
                    yaxis_title='Densidad',
                    showlegend=True
                )
                
                return fig
            except Exception as e:
                logger.error(f"Error en update_distribution: {str(e)}")
                return go.Figure()

        @self.app.callback(
            Output('binary-characteristics', 'figure'),
            Input('patient-id', 'value')
        )
        def update_binary_characteristics(patient_id):
            try:
                binary_vars = ['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking']
                labels = ['Anemia', 'Diabetes', 'Presión Alta', 'Sexo (M)', 'Fumador']
                
                population_props = self.predictor.df[binary_vars].mean()
                patient_values = self.predictor.df.iloc[patient_id][binary_vars]
                
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    name='Población',
                    x=labels,
                    y=population_props,
                    marker_color='lightblue'
                ))
                
                fig.add_trace(go.Scatter(
                    name='Paciente',
                    x=labels,
                    y=patient_values,
                    mode='markers',
                    marker=dict(size=12, color='red')
                ))
                
                fig.update_layout(
                    title='Características binarias: Paciente vs Población',
                    yaxis_title='Proporción',
                    yaxis_range=[0, 1],
                    showlegend=True
                )
                
                return fig
            except Exception as e:
                logger.error(f"Error en update_binary_characteristics: {str(e)}")
                return go.Figure()

        @self.app.callback(
            Output('risk-analysis', 'figure'),
            Input('patient-id', 'value')
        )
        def update_risk_analysis(patient_id):
            try:
                patient = self.predictor.df.iloc[patient_id]
                risk_factors = self.predictor.get_risk_factors(patient)
                
                fig = go.Figure(go.Bar(
                    x=list(risk_factors.keys()),
                    y=list(risk_factors.values()),
                    marker_color=['red' if v == 1 else 'green' for v in risk_factors.values()]
                ))
                
                fig.update_layout(
                    title='Análisis de Factores de Riesgo',
                    yaxis_title='Factor Presente',
                    yaxis_range=[0, 1.2]
                )
                
                return fig
            except Exception as e:
                logger.error(f"Error en update_risk_analysis: {str(e)}")
                return go.Figure()

        @self.app.callback(
            Output('survival-curve', 'figure'),
            Input('patient-id', 'value')
        )
        def update_survival_curve(patient_id):
            try:
                patient_data = self.predictor.df.iloc[patient_id]
                times, patient_survival, base_survival = self.predictor.predict_survival(patient_data)
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=times,
                    y=patient_survival,
                    mode='lines',
                    name=f'Paciente {patient_id}',
                    line=dict(color='red', width=2)
                ))
                
                fig.add_trace(go.Scatter(
                    x=times,
                    y=base_survival,
                    mode='lines',
                    name='Población base',
                    line=dict(color='blue', width=2, dash='dash')
                ))
                
                fig.update_layout(
                    title='Curva de Supervivencia Estimada',
                    xaxis_title='Tiempo (días)',
                    yaxis_title='Probabilidad de supervivencia',
                    yaxis_range=[0, 1],
                    showlegend=True,
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01
                    )
                )
                
                return fig
            except Exception as e:
                logger.error(f"Error en update_survival_curve: {str(e)}")
                return go.Figure()

    def run_server(self, debug: bool = False):
        """Iniciar el servidor de la aplicación."""
        try:
            self.app.run(debug=debug, host="0.0.0.0")
        except Exception as e:
            logger.error(f"Error al iniciar el servidor: {str(e)}")
            raise

if __name__ == '__main__':
    try:
        predictor = HeartFailurePredictor(
            data_path="https://raw.githubusercontent.com/jetabaresj/DespliegueTeam23/refs/heads/main/data/heart_failure_clinical_records_dataset.csv",
            model_path="model.pkl"
        )
        dashboard = DashboardUI(predictor)
        dashboard.run_server(debug=True)
    except Exception as e:
        logger.error(f"Error al iniciar la aplicación: {str(e)}")