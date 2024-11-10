from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Cargar datos
df = pd.read_csv('data/heart_failure_clinical_records_dataset.csv')
df['id'] = df.index  # Añadir columna de ID basada en el índice

app = Dash(__name__)

# Definir el layout
app.layout = html.Div([
    # Título principal
    html.H1('Valorador del riesgo de insuficiencia cardiaca', 
            style={'textAlign': 'center', 'marginBottom': '30px'}),
    
    # Container principal con grid de 2x2
    html.Div([
        # Primera fila
        html.Div([
            # Información del paciente
            html.Div([
                html.H3('Información del paciente'),
                dcc.Dropdown(
                    id='patient-id',
                    options=[{'label': f'Paciente {i}', 'value': i} for i in df.index],
                    value=0,
                    className='input-field'
                ),
                # Panel de información del paciente seleccionado
                html.Div(id='patient-info', className='info-panel')
            ], className='panel'),
            
            # Caracterización de variables numéricas
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
            ], className='panel'),
        ], style={'display': 'flex', 'gap': '20px'}),
        
        # Segunda fila
        html.Div([
            # Características binarias
            html.Div([
                html.H3('Características del paciente vs población'),
                dcc.Graph(id='binary-characteristics')
            ], className='panel'),
            
            # Caracterización según riesgo
            html.Div([
                html.H3('Análisis de riesgo'),
                dcc.Graph(id='risk-analysis')
            ], className='panel'),
        ], style={'display': 'flex', 'gap': '20px'})
    ], style={'display': 'flex', 'flexDirection': 'column', 'gap': '20px'})
], style={'padding': '20px'})

# Callbacks
@callback(
    Output('patient-info', 'children'),
    Input('patient-id', 'value')
)
def update_patient_info(patient_id):
    patient = df.iloc[patient_id]
    return html.Div([
        html.P(f"Edad: {patient['age']} años"),
        html.P(f"Sexo: {'Masculino' if patient['sex'] == 1 else 'Femenino'}"),
        html.P(f"Anemia: {'Sí' if patient['anaemia'] == 1 else 'No'}"),
        html.P(f"Diabetes: {'Sí' if patient['diabetes'] == 1 else 'No'}"),
        html.P(f"Presión Alta: {'Sí' if patient['high_blood_pressure'] == 1 else 'No'}"),
        html.P(f"Fumador: {'Sí' if patient['smoking'] == 1 else 'No'}")
    ])

@callback(
    Output('distribution-plot', 'figure'),
    [Input('variable-selector', 'value'),
     Input('patient-id', 'value')]
)
def update_distribution(selected_var, patient_id):
    patient_value = df.iloc[patient_id][selected_var]
    
    fig = go.Figure()
    
    # Añadir histograma de la población
    fig.add_trace(go.Histogram(
        x=df[selected_var],
        name='Población',
        nbinsx=30,
        histnorm='probability density',
        opacity=0.7
    ))
    
    # Añadir línea vertical para el paciente seleccionado
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

@callback(
    Output('binary-characteristics', 'figure'),
    Input('patient-id', 'value')
)
def update_binary_characteristics(patient_id):
    binary_vars = ['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking']
    labels = ['Anemia', 'Diabetes', 'Presión Alta', 'Sexo (M)', 'Fumador']
    
    # Calcular proporción en la población
    population_props = df[binary_vars].mean()
    
    # Obtener valores del paciente
    patient_values = df.iloc[patient_id][binary_vars]
    
    fig = go.Figure()
    
    # Barras para la población
    fig.add_trace(go.Bar(
        name='Población',
        x=labels,
        y=population_props,
        marker_color='lightblue'
    ))
    
    # Puntos para el paciente
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

@callback(
    Output('risk-analysis', 'figure'),
    Input('patient-id', 'value')
)
def update_risk_analysis(patient_id):
    # Crear un score simplificado basado en factores de riesgo
    patient = df.iloc[patient_id]
    
    # Calcular factores de riesgo (ejemplo simplificado)
    risk_factors = {
        'Edad': 1 if patient['age'] > 60 else 0,
        'Ejección': 1 if patient['ejection_fraction'] < 30 else 0,
        'Creatinina': 1 if patient['serum_creatinine'] > 1.5 else 0,
        'Presión': 1 if patient['high_blood_pressure'] == 1 else 0,
        'Diabetes': 1 if patient['diabetes'] == 1 else 0
    }
    
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

# Añadir CSS personalizado
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>Valorador de Riesgo Cardíaco</title>
        {%favicon%}
        {%css%}
        <style>
            .panel {
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 15px;
                background-color: white;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                flex: 1;
            }
            .input-field {
                margin-bottom: 10px;
            }
            .info-panel {
                margin-top: 15px;
                padding: 10px;
                background-color: #f8f9fa;
                border-radius: 5px;
            }
            .info-panel p {
                margin: 5px 0;
            }
            body {
                background-color: #f5f5f5;
                font-family: Arial, sans-serif;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

if __name__ == '__main__':
    app.run(debug=True)