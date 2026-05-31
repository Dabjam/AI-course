# Отчёт по итоговому проекту

## 1. Паспорт проекта

- **Название:** Прогнозирование оттока клиентов телеком-компании
- **Автор:** Поздышев Эдуард Павлович
- **Группа:** ЭФБО-13-24
- **Контакт:** pozdyshev.e.p@gmail.com

Проект решает задачу предсказания оттока клиентов (churn). Идея простая: у телеком-компании есть база клиентов, и некоторые из них скоро уйдут к конкурентам. Если заранее знать кто — можно предложить им скидку и удержать. Я обучил модель на исторических данных и обернул её в REST API.

---

## 2. Постановка задачи

**Задача:** бинарная классификация — уйдёт клиент или нет.

**Вход:** профиль клиента — 19 признаков (пол, стаж, тип контракта, какими услугами пользуется, сколько платит).

**Выход:** вероятность оттока от 0 до 1 и категория риска (low / medium / high).

**Метрики:**

- **ROC-AUC** — главная метрика, показывает насколько хорошо модель разделяет уходящих и остающихся. 0.5 = случайное угадывание, 1.0 = идеал.
- **F1** — среднее между precision и recall, удобно когда классы несбалансированы.
- **Precision / Recall** — precision: сколько из предсказанных «уйдёт» реально ушли; recall: сколько реально уходящих мы поймали.

---

## 3. Данные

**Источник:** IBM Telco Customer Churn, открытый датасет ([ссылка](https://github.com/IBM/telco-customer-churn-on-icp4d))

**Размер:** 7043 строки, 20 признаков + целевая переменная Churn.

Признаки делятся на:

- числовые: `tenure` (сколько месяцев клиент), `MonthlyCharges`, `TotalCharges`, `SeniorCitizen`
- категориальные: `gender`, `Contract`, `InternetService`, `PaymentMethod` и ещё ~11 признаков про услуги

**Проблемы в данных:**

- `TotalCharges` хранится как строка — пришлось конвертировать в число, 11 строк оказались пустыми → заполнил медианой
- Дисбаланс классов: 73.5% не уходят, 26.5% уходят — не критичный, ничего специального не делал

**Что интересного в EDA:**

- Клиенты с месячным контрактом уходят значительно чаще чем с годовым или двухлетним
- Чем меньше стаж (tenure), тем выше вероятность ухода
- У уходящих клиентов MonthlyCharges в среднем выше

Подробности в `notebooks/01_eda.ipynb`.

---

## 4. Модели

Попробовал три модели:

**Baseline:**

- `DummyClassifier` — просто угадывает случайно с учётом баланса классов. Нужен чтобы показать что наша модель хоть что-то умеет.
- `LogisticRegression` — линейная модель, простая и интерпретируемая.

**Основная:**

- `RandomForestClassifier` — ансамбль деревьев решений. Не требует масштабирования признаков, даёт feature importances.

Параметры RandomForest: 200 деревьев, max_depth=10. Подобрал руками, GridSearchCV не делал из-за времени.

---

## 5. Результаты экспериментов

Разбивка: train 70% / val 10% / test 20%, стратификация по Churn.
Выбор модели по ROC-AUC на val.

**Метрики на val:**

| Модель             | ROC-AUC    | F1         | Precision  | Recall     |
| ------------------ | ---------- | ---------- | ---------- | ---------- |
| Dummy              | 0.4898     | 0.2480     | 0.2500     | 0.2460     |
| LogisticRegression | 0.8543     | 0.6291     | 0.7067     | 0.5668     |
| **RandomForest**   | **0.8559** | **0.5714** | **0.6815** | **0.4920** |

**Метрики RandomForest на тесте:**

| ROC-AUC | F1     | Precision | Recall |
| ------- | ------ | --------- | ------ |
| 0.8389  | 0.5745 | 0.6655    | 0.5053 |

**Почему выбрал RandomForest:**

- Лучший ROC-AUC на val (0.8559 против 0.8543 у LogReg — разница небольшая, но всё же)
- У LogReg лучше F1 и Recall, но ROC-AUC важнее для ранжирования клиентов по риску
- RandomForest не нужно масштабировать признаки
- Можно посмотреть какие признаки важнее (feature importances) — Contract и tenure оказались на первых местах

Подробности в `notebooks/02_experiments.ipynb`.

---

## 6. Сервис

Пайплайн выглядит так:

```
CSV -> preprocess.py -> train.py -> model.pkl
                                     |
POST /predict -> preprocess -> predict_proba -> JSON
```

**API:**

- `GET /health` — возвращает `{"status": "ok", "model_loaded": true}`
- `POST /predict` — принимает JSON с данными клиента, возвращает вероятность и уровень риска

**Стек:** FastAPI + uvicorn, scikit-learn, joblib, pydantic.

---

## 7. Логи и конфигурация

Каждый запрос к `/predict` логируется — вероятность, уровень риска, время ответа.

Параметры (`MODEL_PATH`, порт) вынесены в переменные окружения. Шаблон — `configs/.env.example`. Реальный `.env` в `.gitignore`.

---

## 8. Ограничения

- Модель не переобучается автоматически при новых данных
- Нет авторизации на API
- Гиперпараметры подобраны руками, не через GridSearchCV
- Если клиент сильно отличается от обучающей выборки — предсказание может быть ненадёжным

---

## 9. Тестовый запуск

1. Запустить `uvicorn src.service.app:app --reload` из папки `project/`
2. Открыть `http://localhost:8000/docs`
3. Два запроса через Swagger — скопировать JSON в поле Request body:

   **Высокий риск → ожидаем proba ≈ 0.87, risk_level = high:**

   ```json
   {
     "gender": "Female",
     "SeniorCitizen": 0,
     "Partner": "No",
     "Dependents": "No",
     "tenure": 8,
     "PhoneService": "Yes",
     "MultipleLines": "Yes",
     "InternetService": "Fiber optic",
     "OnlineSecurity": "No",
     "OnlineBackup": "No",
     "DeviceProtection": "Yes",
     "TechSupport": "No",
     "StreamingTV": "Yes",
     "StreamingMovies": "Yes",
     "Contract": "Month-to-month",
     "PaperlessBilling": "Yes",
     "PaymentMethod": "Electronic check",
     "MonthlyCharges": 99.65,
     "TotalCharges": 820.5
   }
   ```

   **Низкий риск → ожидаем proba ≈ 0.02, risk_level = low:**

   ```json
   {
     "gender": "Male",
     "SeniorCitizen": 0,
     "Partner": "No",
     "Dependents": "No",
     "tenure": 34,
     "PhoneService": "Yes",
     "MultipleLines": "No",
     "InternetService": "DSL",
     "OnlineSecurity": "Yes",
     "OnlineBackup": "No",
     "DeviceProtection": "Yes",
     "TechSupport": "No",
     "StreamingTV": "No",
     "StreamingMovies": "No",
     "Contract": "One year",
     "PaperlessBilling": "No",
     "PaymentMethod": "Mailed check",
     "MonthlyCharges": 56.95,
     "TotalCharges": 1889.5
   }
   ```

4. Посмотреть `notebooks/02_experiments.ipynb` — сравнение моделей, ROC-кривые
