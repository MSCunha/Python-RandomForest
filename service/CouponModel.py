from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# =========================================================
#Módulo 2: MODELAGEM E AVALIAÇÃO
# =========================================================
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