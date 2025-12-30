
# Forests of randomized trees

Esse projeto é uma ferramenta interativa desenvolvida para o nosso seminário. Ele usa Machine Learning para prever se um motorista vai aceitar ou não um cupom de desconto baseado no contexto da viagem. A ideia foi criar um dashboard onde a gente consiga filtrar os dados e ver a IA "tomando a decisão" em tempo real.





## Instalação

Para rodar esse projeto, você precisa ter o Python instalado. O gerenciador de pacotes pip vai instalar as dependencias:

```bash
  pip install pandas scikit-learn matplotlib
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

Query Dinâmica: Filtros laterais para testar cenários (ex: motorista indo pra cafeteria).

Fronteira de Decisão (2D): Visualização de como o Random Forest divide o espaço de dados.

Espaço de Decisão (3D): Gráfico com profundidade para explicar sobreposição de dados.

Tendência de Regressão: Visualização de probabilidade gerada pelo Extra Trees.

Log de Performance: Exibição em tempo real dos scores de Validação Cruzada e Tuning


## Documentação do Processo
Esta seção detalha as escolhas técnicas e os processos de engenharia de dados aplicados no desenvolvimento desta ferramenta, servindo como guia para a manutenção e evolução do código.

- Pré-processamento: Label Encoding
O primeiro passo da "limpeza" foi o tratamento de variáveis categóricas. Como o conjunto de dados contém informações textuais (ex: clima, destino, acompanhantes), foi utilizada a classe LabelEncoder do Scikit-Learn.

Funcionamento: Transforma rótulos de texto em valores numéricos inteiros.

Objetivo: Modelos de Machine Learning operam sobre matrizes matemáticas. Sem essa conversão, o algoritmo seria incapaz de realizar os cálculos de distância e probabilidade necessários para os perfis de motorista.

- Random Forest Classifier
Selecionado como o modelo principal de classificação devido à sua robustez.

Funcionamento: É um método de Ensemble Learning que cria uma floresta de 100 árvores de decisão independentes (n_estimators=100). A predição final é obtida através da técnica de votação majoritária.

Objetivo: Ao combinar múltiplos modelos, o sistema reduz drasticamente o risco de overfitting (quando o modelo memoriza o dataset em vez de aprender padrões), garantindo uma generalização superior para novos dados.

- Extra Trees Regressor
Implementado para fornecer uma análise de tendência e probabilidade contínua.

Funcionamento: Diferente da Random Forest, o Extremely Randomized Trees escolhe pontos de corte (splits) de forma estocástica (aleatória) em cada nó.

Objetivo: Esta aleatoriedade ajuda a ignorar o "ruído" estatístico (decisões humanas atípicas presentes no dataset), focando na tendência real de aceitação dos cupons.

- Validação Cruzada (Cross-Validation)Para assegurar a estabilidade estatística do modelo, aplicamos a técnica de K-Fold Cross-Validation com $k=5$.Funcionamento: O código divide o dataset em 5 partes iguais. O modelo é treinado em 4 partes e testado na 5ª, repetindo o ciclo 5 vezes para que cada dado seja testado ao menos uma vez.Objetivo: A média dos resultados exibida na interface é a prova de que a performance do modelo é consistente e não fruto de uma divisão favorável de dados.

- Grid Search (Tuning de Hiperparâmetros)
A otimização do modelo é feita automaticamente através da classe GridSearchCV.

Funcionamento: O sistema executa uma busca exaustiva testando diferentes combinações de profundidade de árvore e número de estimadores.

Objetivo: Sempre que uma nova query é executada, o agente identifica a "receita" de parâmetros que entrega o maior desempenho para aquele cenário específico, garantindo que a IA esteja sempre operando em seu ajuste fino.

## Autores

- [@MSCunha](https://www.github.com/MSCunha)

- [](https://www.github.com/)
## Licença

[MIT](https://choosealicense.com/licenses/mit/)


## Referência

 - [Dataset: In-Vehicle Coupon Recommendation](https://archive.ics.uci.edu/dataset/603/in+vehicle+coupon+recommendation)
 - [Scikit-Learn Documentation](https://scikit-learn.org/stable/user_guide.html)

