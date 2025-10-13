import os

# Caminho base do projeto
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Caminhos para os diretórios de dados
DATA_PATH = os.path.join(PROJECT_ROOT, 'data')
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, r'data/raw/')
PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, r'data/processed')
DADOS_CSV = os.path.join(PROJECT_ROOT, r'data/processed/pof_domicilio.csv')