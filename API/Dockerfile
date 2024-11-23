# Usa una imagen base de Python con herramientas esenciales
FROM python:3.9-slim

# Establece el directorio de trabajo en el contenedor
WORKDIR /app

# Copia los archivos necesarios
COPY api.py .
COPY model.pkl .

# Instala las dependencias necesarias
RUN pip install --no-cache-dir -r requirements.txt

# Exponer el puerto en el que correrá la API
EXPOSE 8000

# Ejecuta la aplicación
CMD ["python", "api.py"]
