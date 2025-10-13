from scripts.ETL import ETL
from config import settings

db_params = {
    'database': '2222120010_Vinicius',
    'user': '2222120010_Vinicius',
    'password': '2222120010_Vinicius',
    'host': 'dataiesb.iesbtech.com.br',  # Endereço do servidor do PostgreSQL
    'port': '5432'       # Porta padrão do PostgreSQL
}

ETL(settings.DADOS_CSV, db_params)