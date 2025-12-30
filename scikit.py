import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.ensemble import RandomForestClassifier, ExtraTreesRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import DecisionBoundaryDisplay

df = pd.read_csv('in-vehicle-coupon-recommendation.csv')
df = df.drop(columns=['car']).fillna(df.mode().iloc[0])

df_num = df.copy()
age_map = {'below21': 20, '21': 21, '26': 26, '31': 31, '36': 36, '41': 41, '46': 46, '50plus': 55}
df_num['age'] = df_num['age'].map(age_map)

#label encoding
encoders = {}
for col in df_num.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df_num[col] = le.fit_transform(df_num[col])
    encoders[col] = le

# GUI
root = tk.Tk()
root.title("Equipe 08 - Análise de Cupom de Veículo")
root.geometry("1200x900")
try: root.state("zoomed")
except: pass

panel = tk.LabelFrame(root, text="Configurações", padx=10, pady=10)
panel.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

def add_menu(label, options):
    tk.Label(panel, text=label, font=("Arial", 9, "bold")).pack(anchor="w")
    cb = ttk.Combobox(panel, values=["Qualquer"] + sorted(list(options.astype(str))), state="readonly")
    cb.set("Qualquer")
    cb.pack(fill=tk.X, pady=(0, 10))
    return cb

f_coffee = add_menu("Frequência Coffee House:", df['CoffeeHouse'].unique())
f_dest = add_menu("Destino:", df['destination'].unique())

tk.Label(panel, text="\n Eixos:", font=("Arial", 9, "bold")).pack(anchor="w")
eixos = ['temperature', 'age', 'income', 'CoffeeHouse', 'Bar']
cb_x = add_menu("Eixo X:", pd.Index(eixos)); cb_x.set("temperature")
cb_y = add_menu("Eixo Y:", pd.Index(eixos)); cb_y.set("age")
cb_z = add_menu("Eixo Z:", pd.Index(eixos)); cb_z.set("income")

# area de Log/Resultados
txt_log = tk.Text(panel, height=15, width=30, font=("Consolas", 8))
txt_log.pack(pady=10)

frame_plot = tk.Frame(root, bg="white")
frame_plot.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

def atualizar():
    d = df_num.copy()
    try:
        if f_coffee.get() != "Qualquer": d = d[df['CoffeeHouse'] == f_coffee.get()]
        if f_dest.get() != "Qualquer": d = d[df['destination'] == f_dest.get()]
    except: pass

    if len(d) < 10:
        messagebox.showwarning("Aviso", "Amostras insuficientes para GridSearch.")
        return

    X = d[[cb_x.get(), cb_y.get()]]
    y = d['Y']

    #random forest classifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42)

    #validacao cruzada
    cv_scores = cross_val_score(rf, X, y, cv=5)

    #grid search Tuning
    param_grid = {'n_estimators': [10, 50], 'max_depth': [None, 5]}
    grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=3)
    grid.fit(X, y)

    #extra tree regressor
    #usada p prever a probabilidade de aceitacao
    et_reg = ExtraTreesRegressor(n_estimators=50, random_state=42)
    et_reg.fit(X, y)

    #atualizar Log
    txt_log.delete(1.0, tk.END)
    txt_log.insert(tk.END, f"--- RESULTADOS ---\n")
    txt_log.insert(tk.END, f"CV Média: {cv_scores.mean():.2f}\n")
    txt_log.insert(tk.END, f"Melhor Param: {grid.best_params_}\n")
    txt_log.insert(tk.END, f"ExtraTrees: {et_reg.score(X, y):.2f}\n")

    #resultado
    for w in frame_plot.winfo_children(): w.destroy()
    fig = plt.figure(figsize=(14, 10))
    plt.rcParams['toolbar'] = 'None'
    
    #plano
    ax1 = fig.add_subplot(221)
    rf.fit(X, y)
    DecisionBoundaryDisplay.from_estimator(rf, X, response_method="predict", cmap=plt.cm.RdYlBu, alpha=0.5, ax=ax1)
    ax1.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, edgecolor='k', cmap=plt.cm.RdYlBu, s=30)
    ax1.set_title("Random Forest")
    ax1.set_xlabel(cb_x.get())
    ax1.set_ylabel(cb_y.get())

    #espaco
    ax2 = fig.add_subplot(222, projection='3d')
    ax2.scatter(d[cb_x.get()], d[cb_y.get()], d[cb_z.get()], c=y, cmap=plt.cm.RdYlBu, edgecolor='k')
    ax2.set_title("Distribuicao espaco")
    ax2.set_xlabel(cb_x.get())
    ax2.set_ylabel(cb_y.get())
    ax2.set_zlabel(cb_z.get())

    ax3 = fig.add_subplot(223)
    #linha de predicao
    ax3.scatter(X.iloc[:, 0], y, alpha=0.3, label="Dados")
    ax3.set_title("Extra Trees: Tendencia de Regressao")
    ax3.set_xlabel(cb_x.get())
    ax3.set_ylabel("Status Aceitação (Y)")

    plt.tight_layout()
    canvas = FigureCanvasTkAgg(fig, master=frame_plot); canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

btn = tk.Button(panel, text="EXECUTAR", command=atualizar, bg="#28a745", fg="white", font=("Arial", 10, "bold"), height=2)
btn.pack(fill=tk.X, pady=20)

atualizar()
root.mainloop()