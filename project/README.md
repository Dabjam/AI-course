# Итоговый проект — Прогнозирование оттока клиентов

## Паспорт проекта

- **Название:** Сервис прогнозирования оттока клиентов (churn prediction)
- **Автор:** Поздышев Эдуард Павлович
- **Группа:** ЭФБО-13-24
- **Контакт:** pozdyshev.e.p@gmail.com

Проект предсказывает, уйдёт ли клиент телеком-компании. Берём данные клиента (стаж, тип контракта, услуги, платежи) и возвращаем вероятность оттока. Данные — открытый датасет IBM Telco Customer Churn.

---

## Структура

```
project/
├── data/telco_churn.csv        # датасет
├── notebooks/
│   ├── 01_eda.ipynb            # разведочный анализ
│   └── 02_experiments.ipynb   # сравнение моделей
├── src/
│   ├── data/preprocess.py      # предобработка
│   ├── models/train.py         # обучение
│   └── service/app.py          # FastAPI
├── configs/
│   ├── config.yaml
│   └── .env.example
├── tests/test_service.py
├── artifacts/model.pkl         # обученная модель
├── Dockerfile
└── requirements.txt
```

---

## Установка

```bash
cd project
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

---

## Запуск

### Обучение модели

```bash
python -m src.models.train
```

Обучит три модели, выведет метрики, сохранит лучшую в `artifacts/model.pkl`.

### Запуск сервиса

```bash
uvicorn src.service.app:app --reload
```

Сервис работает на `http://localhost:8000`.

Эндпоинты:

- `GET /health` — проверка что сервис живой
- `POST /predict` — предсказание по профилю клиента
- `GET /docs` — Swagger UI, можно тестировать прямо в браузере

Пример запроса:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes",
    "Dependents": "No", "tenure": 12, "PhoneService": "Yes",
    "MultipleLines": "No", "InternetService": "Fiber optic",
    "OnlineSecurity": "No", "OnlineBackup": "Yes",
    "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "No", "StreamingMovies": "No",
    "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.35, "TotalCharges": 840.2
  }'
```

Ответ:

```json
{ "churn_probability": 0.2532, "churn_prediction": false, "risk_level": "low" }
```

### Docker

```bash
docker build -t churn-service .
docker run -p 8000:8000 churn-service
```

### Тесты

```bash
pytest tests/ -v
```

---

## Данные

Датасет: IBM Telco Customer Churn, 7043 клиента, 20 признаков.
Лежит в `data/telco_churn.csv` (< 1 МБ, открытые данные).

---

## Тестовый запуск

1. Запустить `uvicorn src.service.app:app --reload`
2. Открыть `http://localhost:8000/docs`
3. Сделать два запроса через Swagger — скопировать JSON в поле Request body:

   **Высокий риск → proba ≈ 0.87, risk_level = high:**

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

   **Низкий риск → proba ≈ 0.02, risk_level = low:**

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

4. Посмотреть `notebooks/02_experiments.ipynb` — таблица сравнения моделей

---

## Ограничения

- Модель обучена на статичном датасете, без переобучения при появлении новых данных
- Нет авторизации на API
