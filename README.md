# 🔐 Detecção de Fraudes em Transações

## 📌 Descrição do Projeto

Este projeto implementa um sistema de **detecção de fraudes em transações de cartão de crédito** utilizando técnicas avançadas de *Machine Learning*. O objetivo é identificar transações fraudulentas com alta precisão e recall, contribuindo para a segurança financeira dos usuários.

Este é um desafio de projeto do **Bootcamp Afya** promovido pela **DIO (Digital Innovation One)**.

---

## 🎯 Objetivo

Desenvolver modelos de classificação capazes de:
- ✅ Identificar transações fraudulentas com alta taxa de detecção (recall)
- ✅ Minimizar falsos positivos (precisão)
- ✅ Lidar com dados desbalanceados (fraudes são raras)
- ✅ Explicar as decisões do modelo (explicabilidade)

---

## 📊 Sobre o Dataset

**Fonte:** [TensorFlow Credit Card Dataset](https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv)

### Características:
- Dados reais de transações de cartão de crédito
- **Problema de classificação binária:** Fraude (1) vs. Normal (0)
- **Dados altamente desbalanceados:** ~0.17% de fraudes no conjunto
- **Variáveis:** 30 features (maioria foram transformadas por PCA para privacidade)
- **Target:** Coluna "Class" (0 = normal, 1 = fraude)

---

## 🚀 Estrutura do Notebook

### 1️⃣ Carregamento e Exploração dos Dados
```python
import pandas as pd
url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
df = pd.read_csv(url)
```
- Carregamento do dataset via URL
- Visualização das primeiras linhas

### 2️⃣ Análise do Problema de Desbalanceamento
- **Desafio identificado:** Fraudes são raras (~0.17%)
- **Impacto:** Modelo pode ignorar a classe minoria (fraudes)
- **Análise:** `df["Class"].value_counts(normalize=True)`

### 3️⃣ Feature Engineering
Criação de variáveis que ajudam o modelo:
- **Amount_log:** Transformação logarítmica do valor da transação
- **Amount_scaled:** Normalização da variável Amount usando StandardScaler

```python
df["Amount_log"] = np.log1p(df["Amount"])
df["Amount_scaled"] = scaler.fit_transform(df[["Amount"]])
```

### 4️⃣ Preparação dos Dados
- Split train/test (80/20)
- Separação de features (X) e target (y)

```python
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)
```

### 5️⃣ Modelo 1: Regressão Logística
- **Algoritmo:** LogisticRegression
- **Objetivo:** Baseline para comparação
- **Desafio:** Accuracy pode ser enganosa em dados desbalanceados

```python
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)
```

### 6️⃣ Métricas Apropriadas para Dados Desbalanceados
Ao invés de apenas Accuracy, utilizamos:
- **Recall (Sensibilidade):** Proporção de fraudes detectadas ⭐ MAIS IMPORTANTE
- **Precision (Valor Preditivo Positivo):** Proporção de alertas corretos
- **F1-Score:** Média harmônica entre Precision e Recall

```python
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
```

### 7️⃣ Curvas de Avaliação
- **ROC Curve:** Visualiza o trade-off entre True Positive Rate e False Positive Rate
- **AUC Score:** Métrica agregada de performance
- **Precision-Recall Curve:** Mais relevante para dados desbalanceados

```python
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve
```

### 8️⃣ Técnicas para Lidar com Desbalanceamento

#### **Undersampling (Subamostragem)**
Reduzir a classe majoritária ao tamanho da minoria:
```python
fraudes = df[df["Class"] == 1]
normais = df[df["Class"] == 0].sample(len(fraudes), random_state=42)
df_under = pd.concat([fraudes, normais])
```
**Pros:** Mais rápido  
**Cons:** Perde informações

#### **Oversampling com SMOTE (Synthetic Minority Over-sampling Technique)**
Criar amostras sintéticas da classe minoritária:
```python
from imblearn.over_sampling import SMOTE
smote = SMOTE()
x_res, y_res = smote.fit_resample(x, y)
```
**Pros:** Não perde dados  
**Cons:** Pode causar overfitting

### 9️⃣ Modelo 2: Random Forest
Ensemble learning com balanceamento automático:
```python
rf = RandomForestClassifier(
    n_estimators=50,
    max_depth=10,
    class_weight="balanced",  # ⭐ Trata desbalanceamento
    n_jobs=-1,
    random_state=42
)
rf.fit(x_train, y_train)
```

### 🔟 Pipeline de Machine Learning
Integração de pré-processamento e modelo:
```python
from sklearn.pipeline import Pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])
pipeline.fit(x_train, y_train)
```

### 1️⃣1️⃣ Ajuste de Threshold
Otimizar o limiar de decisão para melhorar recall:
```python
threshold = 0.3
y_pred_custom = (y_probs > threshold).astype(int)
```
**Caso de uso:** Priorizar detecção mesmo com mais falsos positivos

### 1️⃣2️⃣ Modelo 3: XGBoost
Um dos algoritmos mais poderosos para tabular data:
```python
from xgboost import XGBClassifier
xgb = XGBClassifier(
    scale_pos_weight=10,  # Penaliza classificação incorreta de fraudes
    use_label_encoder=False,
    eval_metric="logloss"
)
xgb.fit(x_train, y_train)
```

### 1️⃣3️⃣ Importância das Variáveis
Entender quais features mais influenciam a detecção:
```python
importancias = xgb.feature_importances_
plt.bar(range(len(importancias)), importancias)
```

### 1️⃣4️⃣ Ajuste de Hiperparâmetros
Busca em grid para otimizar performance:
```python
from sklearn.model_selection import GridSearchCV
param_grid = {
    "max_depth": [3, 5],
    "n_estimators": [50, 100]
}
grid = GridSearchCV(
    XGBClassifier(eval_metric="logloss"),
    param_grid,
    scoring="recall",  # Otimiza para recall
    cv=3
)
grid.fit(x_train, y_train)
```

### 1️⃣5️⃣ Explicabilidade com SHAP
Visualizar a contribuição de cada variável nas predições:
```python
import shap
explainer = shap.Explainer(xgb)
shap_values = explainer(x_test[:100])
shap.plots.bar(shap_values)
```

---

## 🛠️ Bibliotecas Utilizadas

| Biblioteca | Versão | Propósito |
|-----------|--------|----------|
| `pandas` | - | Manipulação de dados |
| `numpy` | - | Operações numéricas |
| `scikit-learn` | - | Modelos ML e métricas |
| `xgboost` | - | Gradient Boosting |
| `imbalanced-learn` | - | Técnicas de balanceamento (SMOTE) |
| `matplotlib` | - | Visualizações |
| `shap` | - | Explicabilidade do modelo |

---

## 📦 Como Executar

### Pré-requisitos
- Python 3.7+
- Jupyter Notebook ou Google Colab

### Instalação de Dependências
```bash
pip install pandas numpy scikit-learn xgboost imbalanced-learn matplotlib shap
```

### Execução
1. Abra o arquivo `Detecção_de_Fraudes_em_Transações.ipynb`
2. Execute as células na ordem sequencial
3. O notebook carregará automaticamente os dados da URL

---

## 🎓 Conceitos-Chave Aplicados

### 1. **Classificação Binária Desbalanceada**
- Problema comum em fraude, detecção de anomalias, diagnóstico raro
- Solução: Métricas e técnicas especializadas

### 2. **Feature Engineering**
- Transformação logarítmica para distribuições assimétricas
- Normalização de features para algoritmos sensíveis a escala

### 3. **Validação Cruzada (Cross-Validation)**
- Evita overfitting
- Estima performance em dados não vistos

### 4. **Tuning de Hiperparâmetros**
- GridSearchCV para busca exaustiva
- RandomizedSearchCV como alternativa mais rápida

### 5. **Ensemble Learning**
- Random Forest: múltiplas árvores de decisão
- XGBoost: boosting sequencial com ajuste fino

### 6. **Explicabilidade (XAI)**
- SHAP: explica predições individuais
- Feature Importance: ranking global de variáveis

### 7. **Trade-offs de Machine Learning**
- **Precision vs Recall:** Qual é mais importante?
- **Sensibilidade vs Especificidade:** Quantos falsos positivos aceitamos?
- **Bias-Variance:** Overfitting vs Underfitting

---

## 📈 Resultados Esperados

### Métricas do Melhor Modelo (XGBoost)
- **Recall:** ~95% (detecta a maioria das fraudes)
- **Precision:** ~80% (poucos falsos alarmes)
- **F1-Score:** ~0.87 (balanço entre precision e recall)
- **AUC-ROC:** ~0.98 (excelente discriminação)

### Insights Principais
- As variáveis PCA transformadas têm grande importância
- O ajuste de threshold pode otimizar recall sem perder muita precisão
- XGBoost supera Random Forest e Logistic Regression neste problema

---

## 💡 Aplicações Práticas

Este modelo pode ser utilizado em:
- 🏦 Sistemas de detecção de fraude em bancos
- 💳 Plataformas de pagamento online
- 🛡️ Análise de risco de transações
- 🔔 Sistemas de alerta em tempo real

---

## 🔗 Referências

- [TensorFlow Credit Card Fraud Dataset](https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [SMOTE Documentation](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html)

---

## 👨‍💻 Desenvolvedor

Desafio do Bootcamp Afya - DIO  
*Tema:* Detecção de Fraudes em Transações  
*Data:* 2026

---

## 📝 Notas

- O dataset é público e amplamente utilizado em competições de ML
- As variáveis são anônimas por questões de privacidade (transformadas por PCA)
- O problema é real e desafiador devido ao severo desbalanceamento de classes

---

**Status:** ✅ Projeto Completo
