"""Assemble a separate, verified hand-in without changing frozen releases."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import zipfile
import importlib.metadata as md
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'delivery_final'
OUT.mkdir(exist_ok=True)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    pred = ROOT/'submissions/final_verified_submission.csv'
    verification = json.loads(pred.with_suffix('.validation.json').read_text())
    assert verification['different_rows'] == 0
    api_log = ROOT/'experiments/reviews/final_api_verified_2026-10-31.log'
    assert 'All 13 unit, integration, reproducibility, and sensitivity tests PASSED' in api_log.read_text(encoding='utf-8', errors='replace')
    shutil.copy2(pred, OUT/'submission.csv')
    scores = json.loads((ROOT/'experiments/model_improvements/five_folds_enhancements_results.json').read_text())['Scale_1M_Dual_0.5']
    # Preserve measured per-fold values, not a manually transcribed comparison.
    pd.DataFrame.from_dict(scores, orient='index').rename_axis('fold').to_csv(OUT/'validation_folds.csv')
    bundle_files = sorted(p for p in (ROOT/'forecast-container/model').rglob('*') if p.is_file())
    manifest = {
        'source_commit': subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
        'selected_model': 'champion_all_stock_ensemble',
        'timestral_included_in_primary_model': False,
        'weights': {'tft': .45, 'lightgbm': .37, 'catboost': .09, 'per_category': .09},
        'submission': verification,
        'api_tests_passed': 13,
        'docker_build_tested': False,
        'model_files_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for p in bundle_files},
        'limitations': ['Local baseline-normalized scores are not leaderboard ranks.', 'Historical folds have been used during development.', 'Timestral mixtures did not establish a robust advantage over the primary model.'],
    }
    (OUT/'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    packages = ['torch','pytorch-lightning','neuralforecast','lightgbm','catboost','pandas','numpy','pyarrow','fastapi','uvicorn','python-multipart','httpx']
    (OUT/'tested_environment.txt').write_text('\n'.join(f'{p}=={md.version(p)}' for p in packages)+'\n', encoding='utf-8')
    guide = '''# Итоговая версия проекта

Для основного соревнования отправить submission.csv. Это проверенный ансамбль;
Timestral не включена в основной прогноз. В архиве лежат код, веса и этот же CSV.

Ансамбль: TFT 45%, LightGBM 37%, CatBoost 9%, категорийные LightGBM 9%.
LightGBM использует два регрессора объёма с разными функциями потерь.
Признаки включают историю продаж, календарь, сетевые продажи и прошлые остатки.
Точный список 28 признаков находится в forecast-container/model/manifest.json.
Категория первого уровня дополнительно служит для разделения категорийных моделей.
Финальная обработка суммы компонентов: умножение на 1.05, порог 0.65,
округление np.rint и ограничение снизу нулём. У компонентов есть своя обработка.

## Повторное получение прогноза

Python 3.11. Установить зависимости из forecast-container/requirements.txt.
Фактические версии среды проверки указаны в tested_environment.txt.
Исходные CSV организаторов передаются отдельно; они не включены в архив.
Команда из корня распакованного архива:

```shell
python generate_submission_release.py --sales sales_dataset_hakaton_train.csv --stock stock_dataset_hakaton_train.csv --output reproduced.csv --reference submission.csv
```

Команда сама отсекает данные после 2026-07-26. Предсказывает 9 недель
2026-07-27 — 2026-09-21, 8789 пар, 79101 строку. Модели загружаются локально,
переобучение и скачивание весов для инференса не требуются.
Повторный прогноз сравнивается с submission.csv строго по ключам.
Источник обучения ансамбля — build_champion_bundle_and_submissions.py;
он включён для изучения, его запуск повторно обучает и заменяет артефакты.

## API и задание со звёздочкой

```shell
cd forecast-container
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

GET /health — состояние; POST /predict — CSV продаж и остатков;
GET /predict — фильтрация по товару/складу; GET /predict/all — весь прогноз.
POST принимает multipart-поля sales и stock. Документация доступна в /docs.
Проверка работающего сервиса: python smoke_test.py http://127.0.0.1:8000
Dockerfile включён. Docker на этом компьютере недоступен: сборка и запуск
контейнера не проверены. Локально выполнены 13 проверок API, включая повторный
инференс, обход кеша и изменение результата при изменении истории.

## Что можно сказать на защите

Мы агрегировали дневные продажи до недель по складу и товару. Дубли остатков
внутри дня складываются, а по неделям берутся отдельные статистики остатков.
Отрицательные остатки допустимы. Прогноз строится сразу на девять недель.
Проверка качества проходит на исторических временных окнах, не случайных строках.
Ансамбль объединяет нейросетевую и табличные модели. Финальный CSV воспроизведён
из исходной истории и сохранённых моделей: все 79101 значения совпали.

Timestral-209M с LoRA улучшила собственный zero-shot результат на двух периодах,
но не показала устойчивого превосходства над ансамблем. Поэтому её не добавляли
в финальную основную модель ради самого факта использования новой нейросети.
Результаты экспериментов — свидетельство проверки гипотез, а не обещание топ-5.

Числа в validation_folds.csv — сохранённые локальные результаты ансамбля.
Они не пересчитаны обучением всех фолдов при сборке архива. Clipped/Unclipped
относительно Mean13 нельзя прямо сравнивать с ранговым score лидерборда.
Временные окна уже использовались при выборе вариантов: это не новый holdout.
'''
    averages = pd.DataFrame.from_dict(scores, orient='index').mean(numeric_only=True)
    guide += '\n## Средние локальные метрики пяти окон\n\n'
    guide += '\n'.join(f'- {k}: {float(averages[k]):.6f}' for k in ['smape','wape','rmse','bias','rmsle','clipped','unclipped']) + '\n'
    (OUT/'RUN_GUIDE.md').write_text(guide, encoding='utf-8')
    files = {}
    def add(path, name=None):
        if path.is_file():
            files[name or path.relative_to(ROOT).as_posix()] = path
    for p in OUT.iterdir():
        if p.suffix not in ('.zip', '.sha256'):
            add(p, p.name)
    for folder in ['forecast-container/app','forecast-container/model','forecast-container/data','forecast-container/tests','sber_hak']:
        for p in (ROOT/folder).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and '.git' not in p.parts:
                if folder != 'sber_hak' or p.suffix == '.py': add(p)
    for p in ROOT.glob('*.py'): add(p)
    for name in ['Dockerfile','requirements.txt','.dockerignore','smoke_test.py','test_api.sh']:
        add(ROOT/'forecast-container'/name)
    add(api_log, 'checks/api_checks.log')
    add(ROOT/'experiments/reviews/final_reproduction_2026-10-31.log', 'checks/reproduction.log')
    add(ROOT/'experiments/reviews/final_delivery_actions_2026-10-31.md', 'checks/actions.md')
    add(ROOT/'experiments/model_improvements/five_folds_enhancements_results.json')
    archive = OUT/'project_final_verified.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=4) as z:
        for name,p in sorted(files.items()): z.write(p,name)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert hashlib.sha256(z.read('submission.csv')).hexdigest() == sha(OUT/'submission.csv')
        for name,digest in manifest['model_files_sha256'].items():
            assert hashlib.sha256(z.read(name)).hexdigest() == digest
    (OUT/'project_final_verified.zip.sha256').write_text(f'{sha(archive)}  {archive.name}\n', encoding='utf-8')
    print(json.dumps({'archive': str(archive), 'bytes': archive.stat().st_size, 'files': len(files), 'submission_sha256': sha(OUT/'submission.csv'), 'checks': 'passed'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
