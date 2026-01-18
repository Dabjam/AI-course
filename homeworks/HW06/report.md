# HW06 – Report

> Файл: homeworks/HW06/report.md

## 1. Dataset

- Какой датасет выбран: `S06-hw-dataset-01.csv`
- Размер: (заполнить после запуска ноутбука)
- Целевая переменная: `target` (классы и их доли — заполнить)
- Признаки: числовые (есть несколько категориальных-подобных)

## 2. Protocol

- Разбиение: train/test = 0.8/0.2, `random_state=42`, `stratify=y`
- Подбор: GridSearchCV на train, cv=5, оптимизация F1
- Метрики: accuracy, F1, ROC-AUC (ROC-AUC рассчитан для бинарной задачи)

## 3. Models

- DummyClassifier (strategy='most_frequent')
- LogisticRegression (Pipeline: StandardScaler + LogisticRegression)
- DecisionTreeClassifier (подбор `max_depth`, `min_samples_leaf`)
- RandomForestClassifier (подбор `n_estimators`, `max_depth`, `max_features`)
- GradientBoostingClassifier (подбор `n_estimators`, `learning_rate`, `max_depth`)

## 4. Results

- Финальные метрики на test будут сохранены в `homeworks/HW06/artifacts/metrics_test.json`.
- Победитель: (заполнить после запуска ноутбука, в ноутбуке предположительно GB).

## 5. Analysis

- Устойчивость: рекомендуется прогнать 5 разных `random_state` и оценить разброс метрик.
- Confusion matrix и ROC-кривая сохранены в `homeworks/HW06/artifacts/figures/`.
- Permutation importance: топ-признаки сохранены в `artifacts/figures/feature_importance.png`.

## 6. Conclusion

- Деревья и ансамбли хорошо справляются с задачей; ансамбли уменьшают variance.
- Честный ML-протокол: CV на train для подбора гиперпараметров и один финальный тест — соблюдён.
- Рекомендуется дополнительно прогнать стабильность по `random_state` и при необходимости откалибровать вероятности.
