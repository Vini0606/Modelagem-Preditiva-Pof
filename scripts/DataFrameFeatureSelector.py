import pandas as pd
import time
from typing import List, Union, Dict, Any

# Importações do Scikit-learn
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SequentialFeatureSelector

class DataFrameFeatureSelector:
    """
    Uma classe para aplicar os métodos de seleção de features Forward Selection
    e Backward Elimination em um DataFrame do pandas.

    Esta classe encapsula o SequentialFeatureSelector do scikit-learn,
    fornecendo uma interface conveniente para trabalhar com dataframes.

    Atributos:
        model (BaseEstimator): O modelo do scikit-learn a ser usado para avaliar as features.
        dataframe (pd.DataFrame): O dataframe completo contendo features e a variável alvo.
        target_column (str): O nome da coluna que serve como variável alvo (y).
        X (pd.DataFrame): As features (X) do dataframe.
        y (pd.Series): A variável alvo (y) do dataframe.
        selected_features_ (List[str]): Lista com os nomes das colunas selecionadas após a execução.
        summary_ (Dict[str, Any]): Dicionário com um resumo da última execução.
    """
    def __init__(self, model: BaseEstimator, dataframe: pd.DataFrame, target_column: str):
        """
        Inicializa o seletor de features.

        Args:
            model (BaseEstimator): Um estimador do scikit-learn (ex: LogisticRegression(), LinearRegression()).
            dataframe (pd.DataFrame): O dataframe de entrada.
            target_column (str): O nome da coluna alvo.
        """
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("O argumento 'dataframe' deve ser um DataFrame do pandas.")
        if target_column not in dataframe.columns:
            raise ValueError(f"A coluna alvo '{target_column}' não foi encontrada no dataframe.")

        self.model = model
        self.dataframe = dataframe
        self.target_column = target_column
        
        # Separa as features (X) e o alvo (y)
        self.y = self.dataframe[self.target_column]
        self.X = self.dataframe.drop(columns=[self.target_column])
        
        # Atributos que serão preenchidos após a execução
        self.sfs_selector_ = None
        self.selected_features_: List[str] = []
        self.summary_: Dict[str, Any] = {}

    def run(self, 
            method: str = 'forward', 
            n_features_to_select: Union[int, str] = 'auto', 
            scoring: str = 'accuracy', 
            cv: int = 5, 
            n_jobs: int = -1) -> 'DataFrameFeatureSelector':
        """
        Executa o processo de seleção de features.

        Args:
            method (str): O método a ser usado. 'forward' ou 'backward'.
            n_features_to_select (Union[int, str]): O número de features a selecionar. 
                                                     Pode ser um inteiro ou 'auto'.
            scoring (str): A métrica de pontuação para avaliar as features (ex: 'accuracy', 'r2', 'f1').
            cv (int): O número de folds para a validação cruzada.
            n_jobs (int): O número de processadores a serem usados (-1 para usar todos).

        Returns:
            DataFrameFeatureSelector: Retorna a própria instância da classe para permitir encadeamento de métodos.
        """
        if method not in ['forward', 'backward']:
            raise ValueError("O método deve ser 'forward' ou 'backward'.")

        print(f"--- Iniciando Seleção de Features ({method.capitalize()}) ---")
        start_time = time.time()

        self.sfs_selector_ = SequentialFeatureSelector(
            self.model,
            n_features_to_select=n_features_to_select,
            direction=method,
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs
        )

        self.sfs_selector_.fit(self.X, self.y)
        
        duration = time.time() - start_time
        print(f"--- Concluído em {duration:.2f} segundos ---")

        # Armazena os resultados
        self.selected_features_ = list(self.X.columns[self.sfs_selector_.get_support()])
        
        self.summary_ = {
            'method': method,
            'scoring_metric': scoring,
            'initial_features_count': self.X.shape[1],
            'features_selected_count': len(self.selected_features_),
            'selected_features_list': self.selected_features_,
            'duration_seconds': round(duration, 2)
        }
        
        print(f"\nResumo da Execução:")
        print(f"  - Features iniciais: {self.summary_['initial_features_count']}")
        print(f"  - Features selecionadas: {self.summary_['features_selected_count']}")
        print(f"  - Lista de features: {self.summary_['selected_features_list']}\n")
        
        return self

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Filtra o dataframe fornecido, mantendo apenas as features selecionadas.

        Args:
            dataframe (pd.DataFrame): O dataframe a ser transformado (pode ser o de treino ou um novo).

        Returns:
            pd.DataFrame: Um novo dataframe contendo apenas as colunas de features selecionadas.
        """
        if not self.selected_features_:
            raise RuntimeError("Você deve executar o método .run() antes de transformar os dados.")
        
        # Retorna um novo dataframe com apenas as colunas selecionadas
        return dataframe[self.selected_features_]