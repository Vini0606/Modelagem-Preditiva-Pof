import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.exceptions import NotFittedError

class PCATransformer:
    """
    Encapsula o processo de padronização e transformação de dados em 
    seus componentes principais.

    Esta classe segue a API do Scikit-learn (com os métodos fit, transform
    e fit_transform) para ser facilmente integrada em pipelines.

    Parâmetros:
    ----------
    n_components : int, float ou None, default=0.95
        Define como selecionar os componentes principais.
        - Se int: O número exato de componentes a serem mantidos.
        - Se float (entre 0.0 e 1.0): A quantidade mínima de variância 
          explicada que os componentes selecionados devem ter.
        - Se None: Todos os componentes são mantidos.
    """
    def __init__(self, n_components=0.95):
        if not ((isinstance(n_components, float) and 0 < n_components < 1) or 
                isinstance(n_components, int) or n_components is None):
            raise ValueError("n_components deve ser um int, um float entre 0 e 1, ou None.")
            
        self.n_components = n_components
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=self.n_components)
        self.feature_names_in_ = None
        self.n_components_ = None
        
    def fit(self, X, y=None):
        """
        Aprende os parâmetros de padronização e PCA a partir dos dados.

        Parâmetros:
        ----------
        X : pd.DataFrame
            O DataFrame de treino com as variáveis a serem transformadas.
        y : Ignorado
            Não é utilizado, presente para compatibilidade com a API do Scikit-learn.
        
        Retorna:
        -------
        self : object
            Retorna a própria instância da classe.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X deve ser um pandas DataFrame.")
        
        self.feature_names_in_ = X.columns.tolist()
        
        # Aprende os parâmetros de padronização e PCA
        X_scaled = self.scaler.fit_transform(X)
        self.pca.fit(X_scaled)
        self.n_components_ = self.pca.n_components_
        
        return self

    def transform(self, X):
        """
        Aplica a padronização e a transformação PCA aprendidas.

        Parâmetros:
        ----------
        X : pd.DataFrame
            O DataFrame a ser transformado.

        Retorna:
        -------
        X_pca : pd.DataFrame
            Um novo DataFrame com os componentes principais como colunas.
        """
        if self.feature_names_in_ is None:
            raise NotFittedError("Esta instância de PCATransformer não foi treinada. Chame 'fit' primeiro.")
        
        # Garante que as colunas de X são as mesmas do treino
        X_reordered = X[self.feature_names_in_]
        
        # Aplica as transformações aprendidas
        X_scaled = self.scaler.transform(X_reordered)
        X_pca_np = self.pca.transform(X_scaled)
        
        # Cria um DataFrame com nomes de colunas informativos
        pc_names = [f"PC_{i+1}" for i in range(self.n_components_)]
        X_pca = pd.DataFrame(X_pca_np, index=X.index, columns=pc_names)
        
        return X_pca
    
    def fit_transform(self, X, y=None):
        """
        Aprende com os dados e os transforma em uma única etapa.

        Parâmetros:
        ----------
        X : pd.DataFrame
            O DataFrame de treino a ser transformado.
        y : Ignorado
            Não é utilizado.
        
        Retorna:
        -------
        X_pca : pd.DataFrame
            Um novo DataFrame com os componentes principais como colunas.
        """
        self.fit(X)
        return self.transform(X)

    def summary(self):
        """Imprime um resumo informativo da transformação."""
        if self.n_components_ is None:
            raise NotFittedError("Esta instância de PCATransformer não foi treinada. Chame 'fit' primeiro.")
        
        total_variance = np.sum(self.pca.explained_variance_ratio_)
        
        print("="*45)
        print("Resumo do PCATransformer")
        print("="*45)
        print(f"Número de features originais : {len(self.feature_names_in_)}")
        print(f"Número de componentes retidos: {self.n_components_}")
        print(f"Variância total explicada    : {total_variance:.4f} ({total_variance:.2%})")
        print("="*45)
        
    def get_loadings(self):
        """
        Retorna as "cargas" (loadings) de cada feature original em cada 
        componente principal. Essencial para a interpretação.
        
        Retorna:
        -------
        loadings_df : pd.DataFrame
            DataFrame com componentes como linhas e features como colunas.
        """
        if self.n_components_ is None:
            raise NotFittedError("Esta instância de PCATransformer não foi treinada. Chame 'fit' primeiro.")
        
        pc_names = [f"PC_{i+1}" for i in range(self.n_components_)]
        loadings_df = pd.DataFrame(
            self.pca.components_, 
            columns=self.feature_names_in_, 
            index=pc_names
        )
        return loadings_df


# --- EXEMPLO DE USO ---
if __name__ == '__main__':
    # 1. Criar um DataFrame de exemplo
    np.random.seed(42)
    data = {
        'area_m2': np.random.uniform(50, 200, 100),
        'qualidade_acabamento': np.random.uniform(1, 10, 100),
        'idade_anos': np.random.uniform(0, 40, 100),
        'feature_ruido': np.random.randn(100) * 5
    }
    # Criar features correlacionadas de propósito
    data['num_quartos'] = (data['area_m2'] / 45 + np.random.normal(0, 0.5, 100)).clip(1)
    data['num_banheiros'] = (data['area_m2'] / 60 + np.random.normal(0, 0.5, 100)).clip(1)
    
    df = pd.DataFrame(data)
    
    # 2. Instanciar o transformador
    # Objetivo: reter componentes que expliquem 90% da variância
    pca_transformer = PCATransformer(n_components=0.90)

    # 3. "Treinar" o transformador e transformar os dados em uma etapa
    df_pca = pca_transformer.fit_transform(df)

    # 4. Ver os resultados
    print("--- DataFrame Original (primeiras 5 linhas) ---")
    print(df.head())
    
    print("\n--- DataFrame Transformado em Componentes Principais (primeiras 5 linhas) ---")
    print(df_pca.head())
    
    # 5. Inspecionar o que foi aprendido
    print("\n--- Resumo da Transformação ---")
    pca_transformer.summary()
    
    print("\n--- Cargas (Loadings) dos Componentes ---")
    print("Mostra como cada feature original contribui para cada componente:")
    print(pca_transformer.get_loadings().round(3))