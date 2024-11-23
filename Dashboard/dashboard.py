from dash import Dash, html, dcc, callback, Output, Input
import plotly.graph_objects as go
import numpy as np  
import requests
import logging
import os

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


API_URL = os.getenv("API_URL", "http://localhost:8000/api")

class DashboardUI:
    def __init__(self):
        """Inicializar la interfaz del dashboard."""
        self.app = Dash(__name__)
        self._setup_layout()
        self._setup_callbacks()
        
        # Configuración global para las gráficas
        self.graph_config = {
            'template': 'plotly_dark',
            'paper_bgcolor': 'rgba(0,0,0,0)',
            'plot_bgcolor': 'rgba(0,0,0,0)',
        }
    
    def _setup_layout(self):
        """Configurar el layout del dash"""
        self.app.layout = html.Div([
            # Navbar
            html.Nav([
                html.H1('Valorador del riesgo de insuficiencia cardiaca', className='dashboard-title'),
            ], className='navbar'),
            
            # Main content
            html.Div([
                # PRIMERA FILA
                html.Div([           
                    # Panel de Información del Paciente
                    html.Div([
                        html.Div([
                            html.H3('Información del paciente', className='panel-title'),
                            dcc.Dropdown(
                                id='patient-id',
                                options=[],
                                value=None,  # Cambiado de 0 a None para evitar errores iniciales
                                className='input-field'
                            ),
                            html.Div(id='patient-info', className='info-panel'),
                        ], className='panel-content'),
                        
                        html.Div([
                            html.H3('Características vs Población', className='panel-title'),
                            dcc.Graph(id='binary-characteristics')
                        ], className='panel-content')
                    ], className='dashboard-panel'),
                
                    # Panel de variables numéricas
                    html.Div([
                        html.H3('Variables Numéricas', className='panel-title'),
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
                    ], className='dashboard-panel')      
                ], className='dashboard-row'),
                
                # SEGUNDA FILA
                html.Div([
                    # Panel de análisis de riesgo
                    html.Div([
                        html.H3('Análisis de Factores de Riesgo', className='panel-title'),
                        dcc.Graph(id='risk-analysis')
                    ], className='dashboard-panel'),
                    # Panel de la curva de supervivencia estimada
                    html.Div([
                        html.H3('Curva de Supervivencia Estimada', className='panel-title'),
                        dcc.Graph(id='survival-curve')
                    ], className='dashboard-panel'),
                    html.Div([
                        html.H3('Correlación Características-Mortalidad', className='panel-title'),
                        dcc.Graph(id='correlation-mortality')
                    ], className='dashboard-panel'),
                ], className='dashboard-row')
            ], className='dashboard-content'),
        ], className='dashboard-container')
        
    def _setup_callbacks(self):
        """ Configurar los callbacks del dashboard"""
        @self.app.callback(
            Output('patient-id', 'options'),
            Input('patient-id', 'value')  # Podría ser innecesario este Input
        )      
        def update_patient_options(_):
            try:
                response = requests.get(f"{API_URL}/patients")
                if response.status_code != 200:
                    logger.error(f"Error en la API: {response.status_code}")
                    return []
                    
                data = response.json()
                return [{'label': f'Paciente {i}', 'value': i} 
                        for i in data.get('patient_ids', [])]
            except requests.RequestException as e:
                logger.error(f"Error de conexión: {str(e)}")
                return []
            except Exception as e:
                logger.error(f"Error inesperado en update_patient_options: {str(e)}")
                return []

        @self.app.callback(
            Output('patient-info', 'children'),
            Input('patient-id', 'value')
        )
        def update_patient_info(patient_id):
            if patient_id is None:
                return html.Div("Seleccione un paciente")
                
            try:
                response = requests.get(f"{API_URL}/patients/{patient_id}")
                if response.status_code != 200:
                    return html.Div("Error al cargar información del paciente")
                    
                patient = response.json()
                return html.Div([
                    html.P(f"Edad: {patient.get('age', 'N/A')} años"),
                    html.P(f"Sexo: {'Masculino' if patient.get('sex') == 1 else 'Femenino'}"),
                    html.P(f"Anemia: {'Sí' if patient.get('anaemia') == 1 else 'No'}"),
                    html.P(f"Diabetes: {'Sí' if patient.get('diabetes') == 1 else 'No'}"),
                    html.P(f"Presión Alta: {'Sí' if patient.get('high_blood_pressure') == 1 else 'No'}"),
                    html.P(f"Fumador: {'Sí' if patient.get('smoking') == 1 else 'No'}")
                ])
            except requests.RequestException as e:
                logger.error(f"Error de conexión: {str(e)}")
                return html.Div("Error de conexión con el servidor")
            except Exception as e:
                logger.error(f"Error en update_patient_info: {str(e)}")
                return html.Div("Error al procesar información del paciente")

        @self.app.callback(
            Output('distribution-plot', 'figure'),
            [Input('variable-selector', 'value'),
             Input('patient-id', 'value')]
        )
        def update_distribution(selected_var, patient_id):
            if selected_var is None or patient_id is None:
                return go.Figure()
                
            try:
                # Obtener datos del paciente y estadísticas
                patient_response = requests.get(f"{API_URL}/patients/{patient_id}")
                stats_response = requests.get(f"{API_URL}/population-stats")
                
                if patient_response.status_code != 200 or stats_response.status_code != 200:
                    return go.Figure()
                
                patient_data = patient_response.json()
                stats = stats_response.json()
                
                var_stats = stats['numerical_stats'][selected_var]
                patient_value = patient_data[selected_var]
                
                # Crear distribución normal simulada
                x = np.linspace(var_stats['min'], var_stats['max'], 100)
                y = np.exp(-0.5 * ((x - var_stats['mean']) / var_stats['std']) ** 2)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=x, y=y,
                    fill='tozeroy',
                    name='Población'
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
                    **self.graph_config
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
            if patient_id is None:
                return go.Figure()
                
            try:
                patient_response = requests.get(f"{API_URL}/patients/{patient_id}")
                stats_response = requests.get(f"{API_URL}/population-stats")
                
                if patient_response.status_code != 200 or stats_response.status_code != 200:
                    return go.Figure()
                
                patient_data = patient_response.json()
                pop_stats = stats_response.json()['binary_stats']
                
                binary_vars = ['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking']
                labels = ['Anemia', 'Diabetes', 'Presión Alta', 'Sexo (M)', 'Fumador']
                
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    name='Población',
                    x=labels,
                    y=[pop_stats.get(var, 0) for var in binary_vars],
                    marker_color='lightblue'
                ))
                
                fig.add_trace(go.Scatter(
                    name='Paciente',
                    x=labels,
                    y=[patient_data.get(var, 0) for var in binary_vars],
                    mode='markers',
                    marker=dict(size=12, color='red')
                ))
                
                fig.update_layout(
                    title='Características binarias: Paciente vs Población',
                    yaxis_title='Proporción',
                    yaxis_range=[0, 1],
                    showlegend=True,
                    **self.graph_config
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
            if patient_id is None:
                return go.Figure()
                
            try:
                response = requests.get(f"{API_URL}/predict/{patient_id}")
                if response.status_code != 200:
                    return go.Figure()
                    
                predictions = response.json()
                risk_factors = predictions.get('risk_factors', {})
                
                fig = go.Figure(go.Bar(
                    x=list(risk_factors.keys()),
                    y=list(risk_factors.values()),
                    marker_color=['red' if v == 1 else 'green' for v in risk_factors.values()]
                ))
                
                fig.update_layout(
                    title='Análisis de Factores de Riesgo',
                    yaxis_title='Factor Presente',
                    yaxis_range=[0, 1.2],
                    **self.graph_config
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
            if patient_id is None:
                return go.Figure()
                
            try:
                response = requests.get(f"{API_URL}/predict/{patient_id}")
                if response.status_code != 200:
                    return go.Figure()
                    
                predictions = response.json()
                survival_data = predictions.get('survival_data', {})
                
                if not all(key in survival_data for key in ['times', 'patient_survival', 'base_survival']):
                    return go.Figure()
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=survival_data['times'],
                    y=survival_data['patient_survival'],
                    mode='lines',
                    name=f'Paciente {patient_id}',
                    line=dict(color='red', width=2)
                ))
                
                fig.add_trace(go.Scatter(
                    x=survival_data['times'],
                    y=survival_data['base_survival'],
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
                    **self.graph_config
                )
                
                return fig
            except Exception as e:
                logger.error(f"Error en update_survival_curve: {str(e)}")
                return go.Figure()
            
        @self.app.callback(
            Output('correlation-mortality', 'figure'),
            Input('patient-id', 'value')
        )
        def update_correlation_mortality(_):
            try:
                response = requests.get(f"{API_URL}/mortality-correlation")
                if response.status_code != 200:
                    return go.Figure()
                    
                correlation_data = response.json()
                
                features = ['smoking', 'anaemia', 'high_blood_pressure', 'diabetes', 'sex']
                mortality_rates = correlation_data['mortality_rates']
                
                # Crear figura con dos subplots
                fig = go.Figure()
                
                # Gráfico de barras para tasas de mortalidad
                fig.add_trace(go.Bar(
                    name='Tasa de Mortalidad',
                    x=features,
                    y=[mortality_rates[f] for f in features],
                    marker_color='rgba(219, 64, 82, 0.7)',
                    text=[f'{rate:.1%}' for rate in mortality_rates.values()],
                    textposition='outside'
                ))
                
                # Añadir línea de referencia para la tasa base de mortalidad
                fig.add_hline(
                    y=mortality_rates.get('base_rate', 0),
                    line_dash="dash",
                    line_color="white",
                    annotation_text="Tasa base de mortalidad",
                    annotation_position="bottom right"
                )
                
                # Actualizar layout
                fig.update_layout(
                    title='Tasa de Mortalidad por Característica',
                    xaxis_title='Características',
                    yaxis_title='Tasa de Mortalidad',
                    yaxis_tickformat='.0%',
                    yaxis_range=[0, max(mortality_rates.values()) * 1.2],
                    **self.graph_config,
                    showlegend=False,
                    height=400
                )
                
                return fig
                
            except Exception as e:
                logger.error(f"Error en update_correlation_mortality: {str(e)}")
                return go.Figure()
        
    def run_server(self, debug: bool = False, port: int = 8050):
        try:
            self.app.run(debug=debug, host="0.0.0.0", port=port)
        except Exception as e:
            logger.error(f"Error al iniciar el servidor: {str(e)}")

if __name__ == '__main__':
    dashboard = DashboardUI()
    dashboard.run_server(debug=True)