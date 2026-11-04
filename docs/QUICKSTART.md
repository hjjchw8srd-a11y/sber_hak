# Запуск проекта

Основной поддерживаемый сценарий — инференс сохранённого ансамбля. Обучение
и foundation-модели требуют отдельных данных и окружений.

## Подготовка

Нужен Python 3.11. Из корня репозитория:

```shell
python -m venv .venv
```

В Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

В Linux/macOS:

```shell
source .venv/bin/activate
```

```shell
python -m pip install -r forecast-container/requirements.txt
```

Версии среды, в которой проверялся комплект сдачи, записаны в
[tested_environment.txt](../delivery_final/tested_environment.txt).
Зависимости устанавливаются через интернет; сам основной инференс загружает
веса локально из `forecast-container/model`. Файлы моделей должны быть скачаны
полностью, а не в виде указателей Git LFS.

## Демонстрация API

Из корня репозитория:

```shell
python -m uvicorn app.main:app --app-dir forecast-container --host 127.0.0.1 --port 8000
```

Откройте `http://127.0.0.1:8000/docs`. В `POST /predict` загрузите
`forecast-container/data/demo_sales.csv` и `demo_stock.csv` в поля `sales` и `stock`.
Демоданные проверяют интерфейс; качество полного прогноза по ним не оценивается.

Во втором терминале с тем же окружением:

```shell
python forecast-container/smoke_test.py http://127.0.0.1:8000
```

| Метод | Путь | Назначение |
|---|---|---|
| GET | `/health` | Готовность моделей |
| POST | `/predict` | Прогноз из CSV продаж и остатков |
| GET | `/predict` | Фильтр последнего прогноза по товару/складу |
| GET | `/predict/all` | Полный последний прогноз |

## Полное воспроизведение

Нужны исходные файлы организаторов. Команда отсекает данные после 2026-07-26:

```shell
python generate_submission_release.py --sales sales_dataset_hakaton_train.csv --stock stock_dataset_hakaton_train.csv --output reproduced.csv --reference delivery_final/submission.csv
```

Проверяются формат, полная сетка ключей, конечность значений и точное совпадение
с эталоном. История, отличающаяся от исходной, может закономерно дать другой прогноз.

## Проверки и Docker

```shell
python forecast-container/tests/test_api.py
```

Часть интеграционных тестов использует исходные CSV организаторов; это не
автономный набор тестов для любого чистого клона. Для демо используйте smoke-test.

При установленном Docker:

```shell
docker build -t retail-demand-api forecast-container
docker run --rm -p 8000:8000 retail-demand-api
```

Затем выполните тот же smoke-test. Docker-сборка не была подтверждена в
зафиксированной среде; успешный локальный тест Python не заменяет эту проверку.
На другом оборудовании время работы и численная воспроизводимость требуют проверки.
