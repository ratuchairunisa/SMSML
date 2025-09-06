FROM python:3.9-slim

WORKDIR /app

# Update pip dulu
RUN pip install --upgrade pip

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "modelling.py", "--input", "Membangun_model/processed/processed_data.npz", "--epochs", "30", "--output", "artifacts"]