import pandas as pd
from sklearn.preprocessing import LabelEncoder

# =========================================================
#TRATAMENTO E PREPARAÇÃO DOS DADOS
class DataProcessor:
    """Classe responsável pelo processamento e preparação dos dados"""

    def __init__(self, filepath):
        self.df_original = pd.read_csv(filepath)
        self.df = None
        self.df_num = None
        self.encoders = {}
        self.age_map = {
            'below21': 20, '21': 21, '26': 26, '31': 31,
            '36': 36, '41': 41, '46': 46, '50plus': 55
        }

    def process_data(self):
        """Processa os dados: limpeza, encoding e transformações"""
        #remover coluna 'car' e tratar valores nulos
        self.df = self.df_original.drop(columns=['car']).copy()

        #preencher nulos com moda para cada coluna
        for col in self.df.columns:
            if self.df[col].isnull().sum() > 0:
                self.df[col].fillna(self.df[col].mode()[0], inplace=True)

        #criar cópia numérica
        self.df_num = self.df.copy()

        #mapear idade
        self.df_num['age'] = self.df_num['age'].map(self.age_map)

        #label encoding para variáveis categóricas
        for col in self.df_num.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            self.df_num[col] = le.fit_transform(self.df_num[col])
            self.encoders[col] = le

        return self.df, self.df_num

    def get_unique_values(self, column):
        """Retorna valores únicos de uma coluna"""
        return self.df[column].unique()