import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import numpy as np
import pandas as pd
import datetime as dt

app = dash.Dash(
    __name__,
    meta_tags = [{
        "name": "viewport",
        "content": "width=device-width, initial-scale=1"
    }]
)

app.title = "Survival Analysis"

server = app.server
app.config.supress_callback_expections = True

# Load data from csv

def load_data():
    df = pd.read_csv('data/heart_failure_clinical_records_dataset.csv')
    
    return df

