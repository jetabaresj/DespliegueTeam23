import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn

url = "https://raw.githubusercontent.com/jetabaresj/DespliegueTeam23/refs/heads/main/data/heart_failure_clinical_records_dataset.csv"
data = pd.read_csv(url)

X = data.drop('DEATH_EVENT', axis=1)
y = data['DEATH_EVENT']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Configurar modelos a probar y sus hiperparametros
models_with_params = {
    "Logistic Regression": (LogisticRegression(max_iter=1000), {
        'C': [0.1, 1.0,],
        'solver': ['liblinear',]
    }),
    "Random Forest": (RandomForestClassifier(), {
        'n_estimators': [50, 100,],
        'max_depth': [None, 10, ]
    }),
    "Support Vector Machine": (SVC(), {
        'C': [0.1, 1.0, 10.0],
        'kernel': ['linear', ]
    }),
    "Gradient Boosting": (GradientBoostingClassifier(), {
        'n_estimators': [50, 100,],
        'learning_rate': [0.01, 0.1,],
        'max_depth': [3, 5, 7]
    })
}

# Definir función para entrenar y registrar modelos con MLflow
def train_and_log_model(model_name, model, params, X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name=model_name):
        
        # Busqueda de hiperparámetros 
        grid_search = GridSearchCV(estimator=model, param_grid=params, scoring='f1', cv=5, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        # Mejor modelo
        best_model = grid_search.best_estimator_       
       
        # Hacer predicciones
        y_pred = best_model.predict(X_test)
        
        # Calcular métricas
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Loggear hiperparámetros y métricas
        mlflow.log_param("model_name", model_name)
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        
        # Registrar el modelo en MLflow
        mlflow.sklearn.log_model(best_model, model_name)

# Iterar sobre cada modelo, entrenarlo y registrarlo con MLflow
for model_name, (model, params) in models_with_params.items():
    print(model_name, (params))
    train_and_log_model(model_name, model, params, X_train, X_test, y_train, y_test)

# Cerrar el seguimiento de experimentos
mlflow.end_run()