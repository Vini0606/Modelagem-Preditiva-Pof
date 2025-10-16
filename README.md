# Projeto de Análise Preditiva: Estimativa de Aluguel Residencial (POF 2017-2018)

## 1. Visão Geral do Projeto

Este projeto de Ciência de Dados tem como objetivo principal desenvolver modelos preditivos para estimar o **Valor do Aluguel Estimado** de domicílios, utilizando dados da **Pesquisa de Orçamentos Familiares (POF) 2017-2018** do Instituto Brasileiro de Geografia e Estatística (IBGE).

O projeto está estruturado em quatro fases principais, abrangendo desde a extração e tratamento dos dados até a modelagem preditiva, tanto para a regressão do valor contínuo do aluguel quanto para a classificação da faixa de aluguel.

## 2. Estrutura do Repositório

O projeto é composto por quatro notebooks Jupyter que documentam o fluxo de trabalho e dois scripts Python que encapsulam a lógica de ETL e a seleção de variáveis.

| Arquivo | Tipo | Descrição |
| :--- | :--- | :--- |
| `01_ETL.ipynb` | Notebook | Demonstra a execução do processo de Extração, Transformação e Carga (ETL). |
| `ETL.py` | Script Python | Contém as funções para ler os dados do banco de dados PostgreSQL, realizar a união das tabelas, conversão de tipos e a engenharia de features. |
| `02_AED.ipynb` | Notebook | Realiza a Análise Exploratória de Dados (AED) para entender a distribuição das variáveis e suas relações. |
| `03_Predição_Aluguel_Estimado.ipynb` | Notebook | Implementa modelos de **Regressão** para prever o valor contínuo do aluguel estimado. Inclui a seleção de variáveis por *Forward Selection* e avaliação de modelos. |
| `04_Predição_Alugel_Estimado_Faixas.ipynb` | Notebook | Implementa modelos de **Classificação** para prever a faixa de aluguel estimado (Muito Baixo, Baixo, Médio, Alto, Muito Alto). |
| `DataFrameFeatureSelector.py` | Script Python | Classe utilitária para realizar a Seleção Sequencial de Features (*Sequential Feature Selection*), como *Forward Selection*, utilizando a API do `scikit-learn`. |

## 3. Metodologia

### 3.1. Extração, Transformação e Carga (ETL)

O processo de ETL é detalhado no script `ETL.py` e executado no notebook `01_ETL.ipynb`.

#### 3.1.1. Extração de Dados
Os dados foram extraídos de um banco de dados **PostgreSQL** (schema `POF_2018`) a partir das seguintes *Views*, que representam diferentes módulos da pesquisa POF:
*   `View_Domicilio`
*   `View_Aluguel_Estimado`
*   `View_Rendimento_Trabalho`
*   `View_Caderneta_Coletiva` (Despesa realizada)
*   `View_Despesa_Individual`
*   `View_Despesa_Coletiva`
*   `View_Condições_Vida`
*   `View_Caracteristica_Dieta`

#### 3.1.2. Transformação e Engenharia de Features
1.  **União de Dados:** As *Views* foram unidas em um único DataFrame (`df_pof_domicilio`) utilizando as chaves `cod_upa` e `num_dom`.
2.  **Agregação de Variáveis:** Variáveis de despesa e rendimento foram agregadas por domicílio (soma ou média).
3.  **Conversão de Tipos:** Conversão segura de colunas para tipos numéricos (`Int64` e `float`).
4.  **Tratamento de Valores Ausentes (Missing Values):**
    *   Valores ausentes no `Aluguel Estimado` foram removidos.
    *   Valores ausentes nas colunas de despesa e rendimento foram imputados pela **mediana** agrupada por `Tipo do domicílio`.
    *   `Qtd de banheiros de uso comum` teve valores ausentes preenchidos com 0.
5.  **Tratamento de Outliers:** O método do **Intervalo Interquartil (IQR)** foi aplicado para limitar os *outliers* nas variáveis numéricas, substituindo valores extremos pelo limite superior calculado por $Q3 + 1.5 \times IQR$.
6.  **Criação da Variável Alvo de Classificação:** A variável `Aluguel Estimado (Faixa)` foi criada usando `pd.qcut` para dividir o `Aluguel Estimado` em 5 faixas (quintis):
    *   `1 - Muito Baixo`
    *   `2 - Baixo`
    *   `3 - Médio`
    *   `4 - Alto`
    *   `5 - Muito Alto`

#### 3.1.3. Carga
O DataFrame final, após o tratamento e engenharia de features, é salvo em um arquivo CSV (`settings.DADOS_CSV`) para ser consumido pelas etapas subsequentes de Análise Exploratória e Modelagem.

### 3.2. Análise Exploratória de Dados (AED)

O notebook `02_AED.ipynb` realiza uma análise detalhada dos dados tratados, focando em:
*   **Análise Univariada:** Estatísticas descritivas (média, desvio padrão, quartis) para variáveis numéricas e contagem de frequência para variáveis categóricas.
*   **Análise Bivariada:**
    *   **Variáveis Numéricas:** Análise de correlação de Pearson entre as variáveis numéricas e o `Aluguel Estimado`.
    *   **Variáveis Categóricas:** Análise de associação entre variáveis categóricas e o `Aluguel Estimado (Faixa)` utilizando o teste Qui-Quadrado ($\chi^2$) e o V de Cramer.
*   **Visualizações:** Geração de gráficos de distribuição, *boxplots* e mapas de calor de correlação para identificar padrões e *insights*.

### 3.3. Modelagem Preditiva (Regressão)

O notebook `03_Predição_Aluguel_Estimado.ipynb` foca na previsão do valor contínuo do aluguel.

#### 3.3.1. Pré-processamento
*   **Codificação:** Aplicação de *One-Hot Encoding* nas variáveis categóricas (com `drop_first=True` para evitar multicolinearidade).
*   **Escalonamento:** Aplicação de `StandardScaler` nas variáveis quantitativas e na variável alvo (`Aluguel Estimado`).

#### 3.3.2. Seleção de Variáveis
Foi utilizada a técnica de **Forward Selection** (implementada na classe `DataFrameFeatureSelector`) para selecionar o subconjunto ideal de features que maximiza a métrica de desempenho.

#### 3.3.3. Modelos e Avaliação
Foram testados modelos de regressão, incluindo:
*   **Regressão Linear**
*   **Ridge Regression**
*   **Lasso Regression**
*   **Random Forest Regressor**

A avaliação dos modelos foi realizada utilizando métricas de regressão, como:
*   **R² (Coeficiente de Determinação)**
*   **MSE (Erro Quadrático Médio)**
*   **MAE (Erro Absoluto Médio)**
*   **Cp de Mallows** (para seleção de variáveis e comparação de modelos)

### 3.4. Modelagem Preditiva (Classificação)

O notebook `04_Predição_Alugel_Estimado_Faixas.ipynb` aborda a previsão da faixa de aluguel.

#### 3.4.1. Pré-processamento
O pré-processamento segue a mesma lógica da regressão, mas a variável alvo é a categórica `Aluguel Estimado (Faixa)`.

#### 3.4.2. Modelo e Avaliação
*   **Modelo:** Foi utilizado o algoritmo **Decision Tree Classifier** (Árvore de Decisão) com `max_depth=5`.
*   **Avaliação:** O desempenho do modelo de classificação foi avaliado através de:
    *   **Matriz de Confusão**
    *   **Acurácia**
    *   **Relatório de Classificação** (Precision, Recall, F1-Score)

## 4. Scripts e Classes Auxiliares

### 4.1. `ETL.py`

O script `ETL.py` contém as funções principais para o pipeline de dados:
*   `lerDados(db_params)`: Conecta ao PostgreSQL e extrai os dados das *Views*, realizando a união e renomeação inicial das colunas.
*   `conversoes(df_pof_domicilio)`: Realiza a conversão de tipos de dados para garantir a integridade numérica.
*   `featuresEnginer(df_pof_domicilio)`: Implementa a engenharia de features, tratamento de *missing values* e *outliers*, e a criação da variável de faixa de aluguel.
*   `ETL(output_path, db_params)`: Função principal que orquestra as etapas e salva o resultado em CSV.

### 4.2. `DataFrameFeatureSelector.py`

Esta classe utilitária estende a funcionalidade do `SequentialFeatureSelector` do `scikit-learn` para facilitar a seleção de features diretamente em DataFrames do pandas.
*   **Métodos Suportados:** `forward` (Forward Selection) e `backward` (Backward Elimination).
*   **Funcionalidade:** Permite a seleção de um número específico de features (`n_features_to_select`) com base em uma métrica de pontuação (`scoring`) e validação cruzada (`cv`).
*   **Integração:** Usada no notebook `03_Predição_Aluguel_Estimado.ipynb` para otimizar o conjunto de variáveis preditoras para o modelo de regressão.

## 5. Requisitos e Configuração

Para replicar o projeto, é necessário:
1.  **Acesso ao Banco de Dados:** Ter acesso ao banco de dados PostgreSQL com o schema `POF_2018` e as *Views* correspondentes. As credenciais de acesso são definidas no notebook `01_ETL.ipynb` (exemplo: `database`, `user`, `password`, `host`, `port`).
2.  **Dependências Python:** Instalar as bibliotecas listadas nos notebooks, incluindo:
    *   `pandas`
    *   `numpy`
    *   `matplotlib`
    *   `seaborn`
    *   `scikit-learn`
    *   `psycopg2` (para a etapa de ETL)
    *   `hyperopt` (mencionada nas importações, possivelmente para Otimização de Hiperparâmetros - HPO)

O ambiente de desenvolvimento sugere o uso de um ambiente virtual Python (v3.13.5, conforme metadados) e a execução via Jupyter Notebooks.

## 6. Conclusão

O projeto demonstra um pipeline completo de Ciência de Dados, desde a ingestão de dados de uma fonte relacional (PostgreSQL) até a aplicação de técnicas avançadas de pré-processamento, engenharia de features, análise exploratória e modelagem preditiva (Regressão e Classificação). A utilização de classes auxiliares como `DataFrameFeatureSelector` indica uma preocupação com a modularidade e a reprodutibilidade do código.
