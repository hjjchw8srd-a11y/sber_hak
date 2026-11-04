# API прогноза спроса

Сервис использует сохранённый ансамбль: TFT 45%, LightGBM 37%, CatBoost 9%,
категорийные модели 9%. Точные настройки находятся в `model/manifest.json`
и `model/postprocessing.json`.

- [Установка, демонстрация, Docker и проверки](../docs/QUICKSTART.md)
- [Архитектура и признаки](../docs/ARCHITECTURE.md)
- [Результаты и ограничения](../docs/RESULTS.md)

Из корня репозитория:

```shell
python -m pip install -r forecast-container/requirements.txt
python -m uvicorn app.main:app --app-dir forecast-container --host 127.0.0.1 --port 8000
```

Интерактивная документация: http://127.0.0.1:8000/docs.
Демоданные: `data/demo_sales.csv`, `data/demo_stock.csv`.
Прогноз строится из переданной истории; полные данные организаторов не включены.

Локальные 13 тестов API прошли при фиксации комплекта сдачи.
Dockerfile включён, но успешная сборка Docker в манифесте не подтверждена.

[Старое описание LGBM52](../docs/archive/API_LGBM52_HISTORICAL.md) сохранено
как исторический документ. Его веса 50/30/10/10 и список 24 признаков
не описывают текущий bundle.
