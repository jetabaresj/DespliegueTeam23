# Usar una imagen base de Python con las herramientas necesarias
FROM python:3.9-slim

# Establecer el directorio de trabajo en el contenedor
WORKDIR /app

# Copiar el archivo de requisitos y el código a /app
COPY requirements.txt ./
COPY dashboard.py ./

# Copiar la carpeta assets (si existe) para el CSS y otros recursos estáticos
COPY assets ./assets

# Instalar las dependencias necesarias
RUN pip install --no-cache-dir -r requirements.txt

# Exponer el puerto en el que corre el dashboard
EXPOSE 8050

# Ejecutar la aplicación
CMD ["python", "dashboard.py"]
