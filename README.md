# Forests of randomized trees

Esse projeto é uma ferramenta interativa desenvolvida para o nosso seminário. Ele usa Machine Learning para prever se um motorista vai aceitar ou não um cupom de desconto baseado no contexto da viagem. A ideia foi criar um dashboard onde a gente consiga filtrar os dados e ver a IA "tomando a decisão" em tempo real.

## Instalação

Para rodar esse projeto, você precisa ter o Python instalado. O gerenciador de pacotes pip vai instalar as dependencias:

```bash
  python3 -m venv venv
```

```bash
  source venv/bin/activate
```

```bash
  pip install -r dependences.txt
```

## Rodando localmente

Clone o projeto

```bash
  https://github.com/MSCunha/Python-RandomForest.git
```

Entre no diretório do projeto

```bash
  cd Python-RandomForest
```

Certifique-se de que o arquivo in-vehicle-coupon-recommendation.csv está na mesma pasta e inicie o script:

```bash
python scikit.py
```

## Funcionalidades

- **Query Dinâmica:** Filtros interativos para simular cenários de viagem e observar a decisão do modelo em tempo real.
- **Fronteira de Decisão (2D):** Visualização bidimensional da separação dos dados realizada pelo Random Forest.
- **Espaço de Decisão (3D):** Gráfico tridimensional que evidencia regiões de sobreposição entre as classes.
- **Tendência de Regressão:** Exibição da probabilidade contínua de aceitação gerada pelo Extra Trees.
- **Log de Performance:** Monitoramento dos scores de Validação Cruzada e Grid Search durante a execução.

---

## Documentação do Processo

Esta seção descreve as principais decisões técnicas adotadas no projeto, explicando **como cada etapa funciona** e  **por que ela foi utilizada** , com foco didático e manutenção futura.

### 🔹 1. Pré-processamento — Label Encoding

Antes do treinamento dos modelos, foi necessário tratar as variáveis categóricas do dataset, que contêm informações textuais como clima, destino e acompanhantes.

**Como funciona**

* Utiliza-se o `LabelEncoder` para converter textos em valores numéricos inteiros.
* Cada categoria textual passa a ser representada por um número.

**Por que foi utilizado**

* Algoritmos de Machine Learning trabalham com dados numéricos.
* Árvores de decisão precisam desses valores para realizar os critérios de divisão (splits) durante o treinamento.

---

### 🌳 2. Random Forest Classifier

O Random Forest foi escolhido como o **modelo principal de classificação** do sistema.

**Como funciona**

* É um método de *Ensemble Learning* baseado em múltiplas árvores de decisão.
* O modelo utiliza 100 árvores independentes (`n_estimators = 100`).
* A decisão final é tomada por **votação majoritária** entre as árvores.

**Por que foi utilizado**

* Reduz significativamente o risco de  *overfitting* .
* Garante maior capacidade de generalização para novos dados.
* É robusto para dados reais e ruidosos, como decisões humanas.

---

### 🌲 3. Extra Trees Regressor

Além da classificação, foi utilizado o Extra Trees para analisar **tendências e probabilidades contínuas** de aceitação do cupom.

**Como funciona**

* Semelhante ao Random Forest, porém com maior aleatoriedade.
* Os pontos de corte (splits) são escolhidos de forma estocástica em cada nó.

**Por que foi utilizado**

* A aleatoriedade ajuda a reduzir o impacto de ruídos estatísticos.
* Permite visualizar tendências gerais de aceitação, mesmo com dados sobrepostos.

---

### 🔁 4. Validação Cruzada (Cross-Validation)

Para garantir que o desempenho do modelo seja confiável, foi aplicada a técnica de **K-Fold Cross-Validation** com `k = 5`.

**Como funciona**

* O dataset é dividido em 5 partes.
* Em cada iteração, 4 partes são usadas para treino e 1 para teste.
* O processo se repete até que todas as partes sejam testadas.

**Por que foi utilizada**

* Evita resultados enviesados por uma única divisão de dados.
* A média dos resultados indica a estabilidade real do modelo.

---

### ⚙️ 5. Grid Search — Tuning de Hiperparâmetros

A otimização dos modelos é realizada automaticamente com o `GridSearchCV`.

**Como funciona**

* O sistema testa diferentes combinações de hiperparâmetros, como:
  * profundidade das árvores
  * número de estimadores
* Avalia cada combinação usando validação cruzada.

**Por que foi utilizado**

* Garante que o modelo opere sempre com os melhores parâmetros possíveis.
* Facilita a demonstração didática do impacto dos hiperparâmetros na performance.
* Permite ajustes específicos para diferentes cenários simulados no dashboard.

## Autores

- [@amanda252](https://github.com/amanda252)
- [@gabrielaqueirxz](https://github.com/gabrielaqueirxz)
- [@MSCunha](https://github.com/MSCunha)
- [@MrErykCardoso](https://github.com/MrErykCardoso)
- [@rafaelfreitas1009](https://github.com/rafaelfreitas1009)

## Licença

[MIT](https://choosealicense.com/licenses/mit/)

## Referência

- [Dataset: In-Vehicle Coupon Recommendation](https://archive.ics.uci.edu/dataset/603/in+vehicle+coupon+recommendation)
- [Scikit-Learn Documentation](https://scikit-learn.org/stable/user_guide.html)
