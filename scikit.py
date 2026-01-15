import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.inspection import DecisionBoundaryDisplay
import seaborn as sns

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

#MÓDULO 2: MODELAGEM E AVALIAÇÃO
class CouponModel:
    """Classe responsável pela modelagem e avaliação de ML"""
    
    def __init__(self):
        self.rf_model = None
        self.et_model = None
        self.best_model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.results = {}
        
    def prepare_train_test(self, X, y, test_size=0.25, random_state=42):
        """Divide dados em treino e teste"""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
    def train_random_forest(self, cv_folds=5):
        """Treina Random Forest com validação cruzada"""
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        #validação cruzada
        cv_scores = cross_val_score(
            self.rf_model, self.X_train, self.y_train, 
            cv=cv_folds, scoring='accuracy'
        )
        
        #treinar modelo final
        self.rf_model.fit(self.X_train, self.y_train)
        
        #avaliar no conjunto de teste
        y_pred = self.rf_model.predict(self.X_test)
        
        self.results['rf'] = {
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'test_accuracy': accuracy_score(self.y_test, y_pred),
            'predictions': y_pred,
            'classification_report': classification_report(
                self.y_test, y_pred, output_dict=True
            ),
            'confusion_matrix': confusion_matrix(self.y_test, y_pred),
            'feature_importance': self.rf_model.feature_importances_
        }
        
        return self.results['rf']
    
    def train_extra_trees(self, cv_folds=5):
        """Treina Extra Trees Classifier"""
        self.et_model = ExtraTreesClassifier(n_estimators=100, random_state=42)
        
        #validação cruzada
        cv_scores = cross_val_score(
            self.et_model, self.X_train, self.y_train, 
            cv=cv_folds, scoring='accuracy'
        )
        
        #treinar modelo final
        self.et_model.fit(self.X_train, self.y_train)
        
        #avaliar no conjunto de teste
        y_pred = self.et_model.predict(self.X_test)
        
        self.results['et'] = {
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'test_accuracy': accuracy_score(self.y_test, y_pred),
            'predictions': y_pred,
            'classification_report': classification_report(
                self.y_test, y_pred, output_dict=True
            ),
            'confusion_matrix': confusion_matrix(self.y_test, y_pred),
            'feature_importance': self.et_model.feature_importances_
        }
        
        return self.results['et']
    
    def optimize_hyperparameters(self, model_type='rf', cv_folds=3):
        """Otimiza hiperparâmetros usando GridSearchCV"""
        if model_type == 'rf':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2']
            }
            base_model = RandomForestClassifier(random_state=42)
        else:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            base_model = ExtraTreesClassifier(random_state=42)
        
        grid_search = GridSearchCV(
            base_model, param_grid, cv=cv_folds, 
            scoring='accuracy', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        #salvar melhor modelo
        self.best_model = grid_search.best_estimator_
        
        #avaliar melhor modelo
        y_pred = self.best_model.predict(self.X_test)
        
        self.results['optimized'] = {
            'best_params': grid_search.best_params_,
            'best_cv_score': grid_search.best_score_,
            'test_accuracy': accuracy_score(self.y_test, y_pred),
            'predictions': y_pred,
            'classification_report': classification_report(
                self.y_test, y_pred, output_dict=True
            ),
            'confusion_matrix': confusion_matrix(self.y_test, y_pred),
            'feature_importance': self.best_model.feature_importances_
        }
        
        return self.results['optimized']

#INTERFACE GRÁFICA
class CouponAnalysisGUI:
    """Interface gráfica para análise de cupons"""
    
    def __init__(self, data_processor):
        self.processor = data_processor
        self.model = CouponModel()
        
        #criar janela principal
        self.root = tk.Tk()
        self.root.title("Equipe 08 - Análise Avançada de Cupom de Veículo")
        self.root.geometry("1400x900")
        try: 
            self.root.state("zoomed")
        except: 
            pass
        
        self.setup_ui()
        
    def setup_ui(self):
        """Configura a interface do usuário"""
        #painel de controle
        panel = tk.LabelFrame(self.root, text="Configurações", padx=10, pady=10)
        panel.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        #filtros
        tk.Label(panel, text="Filtros de Dados", font=("Arial", 10, "bold")).pack(anchor="w", pady=(0, 5))
        
        self.f_coffee = self.add_menu(
            panel, "Frequência Coffee House:", 
            self.processor.get_unique_values('CoffeeHouse')
        )
        self.f_dest = self.add_menu(
            panel, "Destino:", 
            self.processor.get_unique_values('destination')
        )
        
        #seleção de eixos
        tk.Label(panel, text="\nEixos para Visualização", font=("Arial", 10, "bold")).pack(anchor="w", pady=(10, 5))
        eixos = ['temperature', 'age', 'income', 'CoffeeHouse', 'Bar', 'RestaurantLessThan20']
        
        self.cb_x = self.add_menu(panel, "Eixo X:", pd.Index(eixos))
        self.cb_x.set("temperature")
        
        self.cb_y = self.add_menu(panel, "Eixo Y:", pd.Index(eixos))
        self.cb_y.set("age")
        
        self.cb_z = self.add_menu(panel, "Eixo Z:", pd.Index(eixos))
        self.cb_z.set("income")
        
        #configurações de modelo
        tk.Label(panel, text="\nConfiguração do Modelo", font=("Arial", 10, "bold")).pack(anchor="w", pady=(10, 5))
        
        self.optimize_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            panel, text="Otimizar Hiperparâmetros (mais lento)", 
            variable=self.optimize_var
        ).pack(anchor="w")
        
        #botão para executar
        self.btn = tk.Button(
            panel, text="EXECUTAR ANÁLISE", command=self.run_analysis, 
            bg="#28a745", fg="white", font=("Arial", 11, "bold"), height=2
        )
        self.btn.pack(fill=tk.X, pady=20)
        
        #área de log
        tk.Label(panel, text="Resultados Detalhados", font=("Arial", 10, "bold")).pack(anchor="w")
        
        scroll = tk.Scrollbar(panel)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.txt_log = tk.Text(panel, height=25, width=35, font=("Consolas", 8), yscrollcommand=scroll.set)
        self.txt_log.pack(pady=5)
        scroll.config(command=self.txt_log.yview)
        
        #frame para gráficos
        self.frame_plot = tk.Frame(self.root, bg="white")
        self.frame_plot.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
    def add_menu(self, parent, label, options):
        """Adiciona um menu dropdown"""
        tk.Label(parent, text=label, font=("Arial", 9)).pack(anchor="w")
        cb = ttk.Combobox(
            parent, values=["Qualquer"] + sorted(list(options.astype(str))), 
            state="readonly"
        )
        cb.set("Qualquer")
        cb.pack(fill=tk.X, pady=(0, 10))
        return cb
    
    def log_results(self, message):
        """Adiciona mensagem ao log"""
        self.txt_log.insert(tk.END, message + "\n")
        self.txt_log.see(tk.END)
        self.root.update()
        
    def run_analysis(self):
        """Executa a análise completa"""
        self.txt_log.delete(1.0, tk.END)
        self.log_results("="*50)
        self.log_results("INICIANDO ANÁLISE...")
        self.log_results("="*50)
        
        #filtrar dados
        d = self.processor.df_num.copy()
        df_original = self.processor.df.copy()
        
        try:
            if self.f_coffee.get() != "Qualquer":
                mask = df_original['CoffeeHouse'] == self.f_coffee.get()
                d = d[mask]
                
            if self.f_dest.get() != "Qualquer":
                mask = df_original['destination'] == self.f_dest.get()
                d = d[mask]
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao filtrar dados: {str(e)}")
            return
        
        if len(d) < 50:
            messagebox.showwarning("Aviso", "Amostras insuficientes para análise confiável (mínimo: 50).")
            return
        
        self.log_results(f"\nTotal de amostras: {len(d)}")
        self.log_results(f"Distribuição da variável alvo:")
        self.log_results(f"  Aceita (1): {(d['Y'] == 1).sum()} ({(d['Y'] == 1).sum()/len(d)*100:.1f}%)")
        self.log_results(f"  Rejeita (0): {(d['Y'] == 0).sum()} ({(d['Y'] == 0).sum()/len(d)*100:.1f}%)")
        
        #preparar features
        X = d[[self.cb_x.get(), self.cb_y.get()]]
        y = d['Y']
        
        #dividir em treino e teste
        self.log_results("\n" + "="*50)
        self.log_results("DIVIDINDO DADOS (75% treino, 25% teste)...")
        self.model.prepare_train_test(X, y)
        self.log_results(f"Treino: {len(self.model.X_train)} amostras")
        self.log_results(f"Teste: {len(self.model.X_test)} amostras")
        
        #treinar Random Forest
        self.log_results("\n" + "="*50)
        self.log_results("TREINANDO RANDOM FOREST...")
        rf_results = self.model.train_random_forest()
        self.log_results(f"CV Accuracy: {rf_results['cv_mean']:.4f} (±{rf_results['cv_std']:.4f})")
        self.log_results(f"Test Accuracy: {rf_results['test_accuracy']:.4f}")
        self.log_results(f"\nPrecision: {rf_results['classification_report']['1']['precision']:.4f}")
        self.log_results(f"Recall: {rf_results['classification_report']['1']['recall']:.4f}")
        self.log_results(f"F1-Score: {rf_results['classification_report']['1']['f1-score']:.4f}")
        
        #treinar Extra Trees
        self.log_results("\n" + "="*50)
        self.log_results("TREINANDO EXTRA TREES CLASSIFIER...")
        et_results = self.model.train_extra_trees()
        self.log_results(f"CV Accuracy: {et_results['cv_mean']:.4f} (±{et_results['cv_std']:.4f})")
        self.log_results(f"Test Accuracy: {et_results['test_accuracy']:.4f}")
        self.log_results(f"\nPrecision: {et_results['classification_report']['1']['precision']:.4f}")
        self.log_results(f"Recall: {et_results['classification_report']['1']['recall']:.4f}")
        self.log_results(f"F1-Score: {et_results['classification_report']['1']['f1-score']:.4f}")
        
        #otimização (opcional)
        if self.optimize_var.get():
            self.log_results("\n" + "="*50)
            self.log_results("OTIMIZANDO HIPERPARÂMETROS...")
            self.log_results("(Isso pode levar alguns minutos...)")
            
            opt_results = self.model.optimize_hyperparameters()
            
            self.log_results(f"\nMelhores parâmetros:")
            for param, value in opt_results['best_params'].items():
                self.log_results(f"  {param}: {value}")
            
            self.log_results(f"\nBest CV Score: {opt_results['best_cv_score']:.4f}")
            self.log_results(f"Test Accuracy: {opt_results['test_accuracy']:.4f}")
            self.log_results(f"\nPrecision: {opt_results['classification_report']['1']['precision']:.4f}")
            self.log_results(f"Recall: {opt_results['classification_report']['1']['recall']:.4f}")
            self.log_results(f"F1-Score: {opt_results['classification_report']['1']['f1-score']:.4f}")
        
        self.log_results("\n" + "="*50)
        self.log_results("GERANDO VISUALIZAÇÕES...")
        
        #gera as visualizações
        self.plot_results(X, y, d)
        
        self.log_results("\n✓ ANÁLISE CONCLUÍDA!")
        
    def plot_results(self, X, y, d):
        """Gera visualizações dos resultados"""
        for w in self.frame_plot.winfo_children():
            w.destroy()
        
        fig = plt.figure(figsize=(16, 10))
        
        #decision boundary - Random Forest
        ax1 = fig.add_subplot(2, 3, 1)
        try:
            DecisionBoundaryDisplay.from_estimator(
                self.model.rf_model, X, response_method="predict", 
                cmap=plt.cm.RdYlBu, alpha=0.4, ax=ax1
            )
        except:
            pass
        scatter = ax1.scatter(
            X.iloc[:, 0], X.iloc[:, 1], c=y, 
            edgecolor='k', cmap=plt.cm.RdYlBu, s=40, alpha=0.7
        )
        ax1.set_title("Random Forest - Decision Boundary", fontsize=11, fontweight='bold')
        ax1.set_xlabel(self.cb_x.get())
        ax1.set_ylabel(self.cb_y.get())
        plt.colorbar(scatter, ax=ax1, label='Aceitação')
        
        #matriz de confusão - Random Forest
        ax2 = fig.add_subplot(2, 3, 2)
        cm = self.model.results['rf']['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2, cbar=False)
        ax2.set_title("Matriz de Confusão - RF", fontsize=11, fontweight='bold')
        ax2.set_xlabel('Predito')
        ax2.set_ylabel('Real')
        
        #feature importance
        ax3 = fig.add_subplot(2, 3, 3)
        features = [self.cb_x.get(), self.cb_y.get()]
        importances = self.model.results['rf']['feature_importance']
        colors = ['#2ecc71', '#3498db']
        bars = ax3.barh(features, importances, color=colors, alpha=0.7, edgecolor='black')
        ax3.set_title("Importância das Features - RF", fontsize=11, fontweight='bold')
        ax3.set_xlabel('Importância')
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax3.text(width, bar.get_y() + bar.get_height()/2, 
                    f'{importances[i]:.3f}', ha='left', va='center', fontsize=9)
        
        #distribuição 3D
        ax4 = fig.add_subplot(2, 3, 4, projection='3d')
        scatter3d = ax4.scatter(
            d[self.cb_x.get()], d[self.cb_y.get()], d[self.cb_z.get()], 
            c=y, cmap=plt.cm.RdYlBu, edgecolor='k', s=30, alpha=0.6
        )
        ax4.set_title("Distribuição no Espaço 3D", fontsize=11, fontweight='bold')
        ax4.set_xlabel(self.cb_x.get())
        ax4.set_ylabel(self.cb_y.get())
        ax4.set_zlabel(self.cb_z.get())
        
        #comparação de modelos
        ax5 = fig.add_subplot(2, 3, 5)
        models = ['Random Forest', 'Extra Trees']
        accuracies = [
            self.model.results['rf']['test_accuracy'],
            self.model.results['et']['test_accuracy']
        ]
        
        if 'optimized' in self.model.results:
            models.append('RF Otimizado')
            accuracies.append(self.model.results['optimized']['test_accuracy'])
        
        colors_bar = ['#3498db', '#e74c3c', '#2ecc71']
        bars = ax5.bar(models, accuracies, color=colors_bar[:len(models)], alpha=0.7, edgecolor='black')
        ax5.set_title("Comparação de Acurácia", fontsize=11, fontweight='bold')
        ax5.set_ylabel('Acurácia')
        ax5.set_ylim([0, 1])
        ax5.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Baseline')
        
        for bar in bars:
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        #distribuição de probabilidades
        ax6 = fig.add_subplot(2, 3, 6)
        y_proba = self.model.rf_model.predict_proba(X)[:, 1]
        ax6.hist([y_proba[y==0], y_proba[y==1]], bins=20, label=['Rejeitou', 'Aceitou'], 
                color=['#e74c3c', '#2ecc71'], alpha=0.7, edgecolor='black')
        ax6.set_title("Distribuição de Probabilidades - RF", fontsize=11, fontweight='bold')
        ax6.set_xlabel('Probabilidade de Aceitação')
        ax6.set_ylabel('Frequência')
        ax6.legend()
        ax6.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='Threshold')
        
        plt.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, master=self.frame_plot)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def run(self):
        """Inicia a aplicação"""
        self.root.mainloop()

#EXECUÇÃO PRINCIPAL
if __name__ == "__main__":
    #processa dados
    processor = DataProcessor('in-vehicle-coupon-recommendation.csv')
    processor.process_data()
    
    #inicia a interface
    app = CouponAnalysisGUI(processor)
    app.run()