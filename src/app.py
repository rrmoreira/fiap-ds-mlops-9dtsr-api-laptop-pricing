"Api para precificação de laptops utilizando Machine Learning"
from datetime import datetime
import json
import boto3
import joblib

model = joblib.load("model/model.pkl")

with open("model/model_metadata.json", "r", encoding="utf-8") as f:
    model_info = json.load(f)
    
cloudwatch = boto3.client("cloudwatch", region_name="us-east-1")

def write_real_data(data, prediction):
    """
    Funcao para escrever os dados consumidos para depois serem estudados 
    para desvios de dados, modelo ou conceito.
    Args:
        data (dict): dicionario de dados com todos atributos do laptop.
        prediction (float): valor de predicao (preço do laptop).
    """
    now = datetime.now()
    now_formatted = now.strftime("%d-%m-%Y %H:%M")
    
    file_name = f"{now.strftime('%Y-%m-%d')}_laptop_prediction_data.csv"
    
    data["price"] = prediction
    data["timestamp"] = now_formatted
    data["model_version"] = model_info["version"]
    
    s3 = boto3.client("s3")
    bucket_name = "fiap-ds-mlops"
    s3_path = "laptop-prediction-real-data"
    
    try:
        existing_object = s3.get_object(Bucket=bucket_name, Key=f'{s3_path}/{file_name}')
        existing_data = existing_object['Body'].read().decode('utf-8').strip().split('\n')
        existing_data.append(','.join(map(str, data.values())))
        updated_content = '\n'.join(existing_data)
    except s3.exceptions.NoSuchKey:
        # Se o arquivo não existir, cria um novo
        updated_content = ','.join(data.keys()) + '\n' + ','.join(map(str, data.values()))
        
    s3.put_object(Body=updated_content, Bucket=bucket_name, Key=f'{s3_path}/{file_name}')

def input_metrics(data, prediction):
    """
    Funcao para escrever metricas customizadas no CloudWatch.
    
    Args:
        data (dict): dicionario de dados com todos atributos do laptop.
        prediction (float): valor de predicao (preço do laptop).
    """
    cloudwatch.put_metric_data(
        MetricData = [
            {
                'MetricName': 'Price Prediction',
                'Value': prediction,
                'Dimensions': [{'Name': "Currency", 'Value': "INR"}]
            },
        ], Namespace='Laptop Pricing Model')
        
    for key, value in data.items():
        cloudwatch.put_metric_data(
            MetricData = [
                {
                    'MetricName': 'Latptop Feature',
                    'Value': 1,
                    'Unit': 'Count',
                    'Dimensions': [{'Name': key, 'Value': str(value)}]
                },
            ], Namespace='Laptop Pricing Features')

def prepare_payload(data):
    """
    Prepara o payload para a API.
    
    Args:
        data (dict): dicionario de dados com todos atributos do laptop.
    
    Returns:
        dict: dicionario com os dados preparados para a API.
    """
    data_processed = []
    
    data_processed.append(int(data["ram_gb"]))
    data_processed.append(int(data["ssd"]))
    data_processed.append(int(data["hdd"]))
    data_processed.append(int(data["graphic_card"]))
    data_processed.append(int(data["warranty"]))
        
    conditions = {
        "brand": {"asus","dell","hp","lenovo","other"},
        "processor_brand": {"amd","intel","m1"},
        "processor_name": {"core i3","core i5","core i7","other","ryzen 5","ryzen 7"},
        "os": {"other","windows"},
        "weight": {"casual","gaming","thinnlight"},
        "touchscreen": {"0","1"},
        "ram_type": {"ddr4","other"},
        "os_bit": {"32","64"}
    }
    
    for key, values in conditions.items():
        for value in values:
            data_processed.append(1 if data[key] == value else 0)
            
    return data_processed