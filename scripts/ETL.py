import psycopg2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f, pearsonr, chi2_contingency
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import math

pd.options.display.float_format = '{:.2f}'.format

def lerDados(db_params):

    def lerDespesaColetiva(db_params):

        try:
            conn = psycopg2.connect(**db_params)
            cursor = conn.cursor()
        except psycopg2.Error as e:
            print(f"Erro ao se conectar ao banco de dados: {e}")

        cursor.execute('SELECT * FROM "POF_2018"."View_Despesa_Coletiva";')
        registros = cursor.fetchall()
        df = pd.DataFrame(registros, columns=[desc[0] for desc in cursor.description])
        
        conn.close()

        return df

    def lerDespesaIndividual(db_params):

        try:
            conn = psycopg2.connect(**db_params)
            cursor = conn.cursor()
        except psycopg2.Error as e:
            print(f"Erro ao se conectar ao banco de dados: {e}")

        cursor.execute('SELECT * FROM "POF_2018"."View_Despesa_Individual";')
        registros = cursor.fetchall()
        df = pd.DataFrame(registros, columns=[desc[0] for desc in cursor.description])
        
        conn.close()

        return df

    def lerCondicoesVida(db_params):

        try:
            conn = psycopg2.connect(**db_params)
            cursor = conn.cursor()
        except psycopg2.Error as e:
            print(f"Erro ao se conectar ao banco de dados: {e}")

        cursor.execute('SELECT * FROM "POF_2018"."View_Condições_Vida";')
        registros = cursor.fetchall()
        df = pd.DataFrame(registros, columns=[desc[0] for desc in cursor.description])
        
        conn.close()

        return df

    def lerCaracteristicaDieta(db_params):

        try:
            conn = psycopg2.connect(**db_params)
            cursor = conn.cursor()
        except psycopg2.Error as e:
            print(f"Erro ao se conectar ao banco de dados: {e}")

        cursor.execute('SELECT * FROM "POF_2018"."View_Caracteristica_Dieta";')
        registros = cursor.fetchall()
        df = pd.DataFrame(registros, columns=[desc[0] for desc in cursor.description])
        
        conn.close()

        return df

    def lerCadernetaColetiva(db_params):

        try:
            conn = psycopg2.connect(**db_params)
            cursor = conn.cursor()
        except psycopg2.Error as e:
            print(f"Erro ao se conectar ao banco de dados: {e}")

        cursor.execute('SELECT * FROM "POF_2018"."View_Caderneta_Coletiva";')
        registros = cursor.fetchall()
        df = pd.DataFrame(registros, columns=[desc[0] for desc in cursor.description])
        
        conn.close()

        return df

    def lerAluguelEstimado(db_params):

        try:
            conn = psycopg2.connect(**db_params)
            cursor = conn.cursor()
        except psycopg2.Error as e:
            print(f"Erro ao se conectar ao banco de dados: {e}")

        cursor.execute('select * FROM "POF_2018"."View_Aluguel_Estimado";')
        registros = cursor.fetchall()
        df = pd.DataFrame(registros, columns=[desc[0] for desc in cursor.description])

        df['v8000'] = df['v8000'].replace(9999999.99, np.nan)
        df['v8000'] = df['v8000'].replace(99999.00, np.nan)
        
        conn.close()

        return df

    def lerDomicilio(db_params):

        try:
            conn = psycopg2.connect(**db_params)
            cursor = conn.cursor()
        except psycopg2.Error as e:
            print(f"Erro ao se conectar ao banco de dados: {e}")

        cursor.execute('select * from "POF_2018"."View_Domicilio";')
        registros = cursor.fetchall()
        pof_domicilio = pd.DataFrame(registros, columns=[desc[0] for desc in cursor.description])

        pof_domicilio.columns = ['cod_upa', 'num_dom', 'uf', 'Estratos do plano amostra', 'Situação do Domicílio',
        'Tipo do domicílio', 'Material das paredes externas', 'Material do telhado',
        'Material do piso', 'Qtd de cômodos', 'Qtd de cômodos dormitórios', 'Forma de abastecimento de água',
        'Frequência da água proveniente de rede geral', 'Tipo de chegada da água', 'A água é aquecida por energia elétrica?',
        'A água é aquecida por gás?', 'A água é aquecida por energia solar?', 'A água é aquecida por lenha ou carvão?', 
        'A água é aquecida por outra forma?', 'Qtd de banheiros exclusivos', 'Qtd de banheiros de uso comum', 
        'Utiliza sanitário ou buraco para dejeções?','Tipo de escoadouro sanitário', 'Destino dado ao lixo', 
        'Energia elétrica é de rede geral?', 'Rede elétrica proveniente de outra origem?',
        'Frequência da energia elétrica de rede geral', 'Utiliza-se gás butijão na preparação de alimentos?', 
        'Utiliza-se lenha ou carvão na preparação de alimentos?', 'Utiliza-se energia elétrica na preparação de alimentos?', 
        'Utiliza-se outro combustível na preparação de alimentos?', 'Este domicílio é:', 'Este contrato de aluguel é:', 
        'A rua onde se localiza é pavimentada?', 'O serviço de correios é realizado:', 'Situação de segurança alimentar']
        
        pof_domicilio['Situação do Domicílio'] = pof_domicilio['Situação do Domicílio'].replace({'1': 'Urbano', '2': 'Rural'})
        
        conn.close()

        return pof_domicilio

    def lerRendimentoTrababalho(db_params):

        try:
            conn = psycopg2.connect(**db_params)
            cursor = conn.cursor()
        except psycopg2.Error as e:
            print(f"Erro ao se conectar ao banco de dados: {e}")

        cursor.execute('select * from "POF_2018"."View_Rendimento_Trabalho";')
        registros = cursor.fetchall()
        pof_rendimento = pd.DataFrame(registros, columns=[desc[0] for desc in cursor.description])
        pof_rendimento['v8500_defla'] = pof_rendimento['v8500_defla'].astype(float)
        pof_rendimento['v531112_defla'] = pof_rendimento['v531112_defla'].astype(float)
        pof_rendimento['v531122_defla'] = pof_rendimento['v531122_defla'].astype(float)
        pof_rendimento['v531132_defla'] = pof_rendimento['v531132_defla'].astype(float)

        pof_rendimento.columns = ['cod_upa', 'num_dom', 'cod_informante', 'quadro', 'sub_quadro', 'seq', 'produto',
                    'Nesse trabalho o informante era:',
                    'Tipo de trabalhador não remunerado em ajuda a membro do domicílio ou parente.',
                    'era servidor público estatutário (federal, estadual, municipal)?',
                    'tinha carteira de trabalho assinada?',
                    'era contribuinte de instituto de previdência?',
                    ' forma de pagamento do último rendimento bruto mensal recebido nesse trabalho',
                    'Valor em reais (R$) do rendimento bruto',
                    'Valor em reais (R$) da dedução com previdência pública',
                    'Valor em reais (R$) da dedução com imposto de renda',
                    'Valor em reais (R$) da dedução com iss e outros impostos',
                    'Último mês que o rendimento foi recebido',
                    'Número de meses que o rendimento foi recebido',
                    'Quantas horas trabalhava normalmente, por semana',
                    'Qual a duração habitual do deslocamento para esse trabalho?',
                    'deflator', 'Valor do rendimento deflacionado',
                    'Valor da dedução com previdência pública deflacionado',
                    'Valor da dedução com imposto de renda deflacionado',
                    'Valor da dedução com iss e outros impostos deflacionado',
                    'valor_imputado', 'fator_anualizacao',
                    'o cargo, a função, a profissão ou o ofício que a pessoa exercia habitualmente no trabalho', 'denominacao_atividade_cnae']

        conn.close()

        return pof_rendimento

    df_pof_rendimentos = lerRendimentoTrababalho(db_params)
    df_pof_domicilio =  lerDomicilio(db_params)
    df_aluguel_estimado = lerAluguelEstimado(db_params)
    df_caderneta_coletiva = lerCadernetaColetiva(db_params)
    df_caracte_dieta = lerCaracteristicaDieta(db_params)
    df_condicoes_vida = lerCondicoesVida(db_params)
    df_despesa_individual = lerDespesaIndividual(db_params)
    df_despesa_coletiva = lerDespesaColetiva(db_params)

    # 1. Aluguel Estimado
    df_pof_domicilio = pd.merge(
            df_pof_domicilio,
            df_aluguel_estimado[['cod_upa', 'num_dom', 'v8000']],
            how='left',
            on=['cod_upa', 'num_dom']
    ).rename(columns={'v8000': 'Aluguel Estimado'})

    del df_aluguel_estimado
    
    # 2. Caderneta Coletiva
    df_pof_domicilio = pd.merge(
        df_pof_domicilio,
        df_caderneta_coletiva.groupby(['cod_upa', 'num_dom'])['v8000_defla'].sum(),
        how='left',
        on=['cod_upa', 'num_dom']
    ).rename(columns={'v8000_defla': 'Valor em reais (R$) de despesa realizada'})

    del df_caderneta_coletiva

    # 3. Condições de Vida
    df_pof_domicilio = pd.merge(
        df_pof_domicilio,
        df_condicoes_vida.groupby(['cod_upa', 'num_dom'])[['v6102', 'v6103']].mean(),
        how='left',
        on=['cod_upa', 'num_dom']
    ).rename(columns={'v6102': 'Rendimento mensal mínimo geral (R$)', 
                        'v6103': 'Rendimento mensal mínimo p\\ alimentação (R$)'})
    
    del df_condicoes_vida

    # 4. Despesa Individual
    df_pof_domicilio = pd.merge(
        df_pof_domicilio,
            df_despesa_individual.groupby(['cod_upa', 'num_dom'])['v8000_defla'].sum(),
            how='left',
            on=['cod_upa', 'num_dom']
        ).rename(columns={'v8000_defla': 'Valor em reais (R$) de despesa individual'})

    del df_despesa_individual

    # 5. Despesa Coletiva
    df_pof_domicilio = pd.merge(
        df_pof_domicilio,
        df_despesa_coletiva.groupby(['cod_upa', 'num_dom'])['v8000_defla'].sum(),
        how='left',
        on=['cod_upa', 'num_dom']
    ).rename(columns={'v8000_defla': 'Valor em reais (R$) de despesa coletiva'})

    del df_despesa_coletiva
    
    cols = ['Valor em reais (R$) do rendimento bruto',
        'Valor em reais (R$) da dedução com previdência pública',
        'Valor em reais (R$) da dedução com imposto de renda',
        'Valor em reais (R$) da dedução com iss e outros impostos']

    df_pof_domicilio = pd.merge(
            df_pof_domicilio,
            df_pof_rendimentos.groupby(['cod_upa', 'num_dom'])[cols].sum(),
            how='left',
            on=['cod_upa', 'num_dom']
        )

    del df_pof_rendimentos
    
    return df_pof_domicilio 

def conversoes(df_pof_domicilio):

    # 1. Crie listas com os nomes das colunas para cada tipo
    colunas_para_int = [
            'Qtd de cômodos',
            'Qtd de cômodos dormitórios',
            'Qtd de banheiros exclusivos',
            'Qtd de banheiros de uso comum'
        ]

    colunas_para_float = [
            'Valor em reais (R$) do rendimento bruto',
            'Valor em reais (R$) da dedução com previdência pública',
            'Valor em reais (R$) da dedução com imposto de renda',
            'Valor em reais (R$) da dedução com iss e outros impostos',
            'Valor em reais (R$) de despesa realizada',
            'Valor em reais (R$) de despesa individual',
            'Valor em reais (R$) de despesa coletiva',
            'Aluguel Estimado',
            'Rendimento mensal mínimo geral (R$)',
            'Rendimento mensal mínimo p\\ alimentação (R$)'
        ]

        # 2. Converta as colunas de forma segura usando um loop
        # Para colunas de inteiros

    for col in colunas_para_int:
        # Primeiro, converte para numérico, transformando erros em Nulos (NaN)
        temp_col = pd.to_numeric(df_pof_domicilio[col], errors='coerce')
        # Depois, converte para o tipo Int64, que suporta nulos
        df_pof_domicilio[col] = temp_col.astype('Int64')

    # Para colunas de float
    for col in colunas_para_float:
        # Apenas converter para numérico é suficiente, pois float já suporta NaN
        df_pof_domicilio[col] = pd.to_numeric(df_pof_domicilio[col], errors='coerce')

    return df_pof_domicilio

def featuresEnginer(df_pof_domicilio):

    df_pof_domicilio['Aluguel Estimado'] = df_pof_domicilio['Aluguel Estimado'].replace(9999999.99, np.nan)

    df_pof_domicilio = df_pof_domicilio.dropna(subset=['Aluguel Estimado'])
    
    nomes_faixas = ['1 - Muito Baixo', '2 - Baixo', '3 - Médio', '4 - Alto', '5 - Muito Alto']
    df_pof_domicilio['Aluguel Estimado (Faixa)'] = pd.qcut(df_pof_domicilio['Aluguel Estimado'], q=5, labels=nomes_faixas, duplicates='drop')
    df_pof_domicilio['Aluguel Estimado (Faixa)'] = df_pof_domicilio['Aluguel Estimado (Faixa)'].astype(str)

    df_pof_domicilio['Qtd de banheiros de uso comum'] = df_pof_domicilio['Qtd de banheiros de uso comum'].fillna(0)
    df_pof_domicilio['Valor em reais (R$) de despesa realizada'] = df_pof_domicilio.groupby('Tipo do domicílio')['Valor em reais (R$) de despesa realizada'].transform(lambda x: x.fillna(x.median()))
    df_pof_domicilio['Valor em reais (R$) do rendimento bruto'] = df_pof_domicilio.groupby('Tipo do domicílio')['Valor em reais (R$) do rendimento bruto'].transform(lambda x: x.fillna(x.median()))
    df_pof_domicilio = df_pof_domicilio.dropna(subset=['Aluguel Estimado', 
                                                    'Valor em reais (R$) de despesa individual', 
                                                    'Valor em reais (R$) de despesa coletiva'])
    
    df_pof_domicilio.drop(labels=['Valor em reais (R$) da dedução com previdência pública',
                                'Valor em reais (R$) da dedução com imposto de renda',
                                'Valor em reais (R$) da dedução com iss e outros impostos'], axis=1, inplace=True)


    # Loop para tratar cada variável
    for variavel in df_pof_domicilio.select_dtypes(include=np.number).columns:
        
        # 4. Calcular Q1 (25º percentil) e Q3 (75º percentil)
        Q1 = df_pof_domicilio[variavel].quantile(0.25)
        Q3 = df_pof_domicilio[variavel].quantile(0.75)

        # 5. Calcular o IQR
        IQR = Q3 - Q1

        # 6. Calcular o Limite Superior para outliers
        limite_superior = Q3 + (1.5 * IQR)
        limite_superior = math.ceil(limite_superior)

        # Substituir outliers pelo limite superior
        df_pof_domicilio.loc[df_pof_domicilio[variavel] > limite_superior, variavel] = limite_superior
    
    return df_pof_domicilio

def ETL(output_path, db_params):

    df_pof_domicilio = lerDados(db_params)
    df_pof_domicilio = conversoes(df_pof_domicilio)
    df_pof_domicilio = featuresEnginer(df_pof_domicilio)
    df_pof_domicilio.to_csv(output_path, index=False)

