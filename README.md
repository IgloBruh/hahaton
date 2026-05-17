# Прогноз почасовой выработки ВЭС

Решение задачи прогнозирования почасовой генерации ветроэлектростанции мощностью **90,09 МВт** (26 турбин Siemens Gamesa SG 3.4-132, координаты 46,83° с.ш. 38,72° в.д., высота ротора 80 м) на основе метеопрогноза.

## Структура проекта

```
.
├── config.py                  # пути, гиперпараметры, ENSEMBLE_CONFIGS
├── train.py                   # обучение LightGBM-ансамбля → models/model.pkl
├── tune.py                    # Optuna-поиск гиперпараметров LightGBM
├── inference.py               # инференс LightGBM → predictions.csv
├── train_autogluon.py         # обучение AutoGluon → models/autogluon_predictor/
├── inference_autogluon.py     # инференс AutoGluon → predictions_autogluon.csv
├── requirements.txt
├── data/
│   ├── train_dataset.csv      # обучающая выборка (с целевым столбцом)
│   └── valid_features.csv     # тестовая выборка (без цели, для сабмита)
├── models/
│   ├── model.pkl              # LightGBM-ансамбль (создаётся train.py)
│   └── autogluon_predictor/   # AutoGluon-предиктор (создаётся train_autogluon.py)
└── src/
    ├── data_loader.py         # чтение CSV
    ├── preprocessing.py       # очистка + 190 признаков
    ├── model.py               # LightGBM-ансамбль
    └── utils.py               # сидирование, метрики
```

## Установка

В проекте два окружения: основное (Python 3.14) и отдельное для AutoGluon (Python 3.11),
потому что AutoGluon не поддерживает Python 3.12+.

### Основное окружение — LightGBM-пайплайн

```bash
py -3.14 -m venv .venv
.venv\Scripts\pip install lightgbm optuna pandas numpy scikit-learn joblib matplotlib
```

### AutoGluon-окружение

```bash
py -3.11 -m venv .venv_ag
.venv_ag\Scripts\pip install "autogluon.tabular[all]" lightgbm pandas numpy scikit-learn joblib
```

## Запуск

### Пайплайн 1 — LightGBM-ансамбль

```bash
# Шаг 1 (опционально): подобрать гиперпараметры через Optuna (50 проб, ~15 мин)
.venv\Scripts\python tune.py

# Шаг 2: обучить ансамбль из 7 членов (~5–10 мин)
# Также генерирует pi_scores.json — нужен для PI-фильтрации в AutoGluon
.venv\Scripts\python train.py

# Шаг 3: сгенерировать predictions.csv
.venv\Scripts\python inference.py
```

> Если `tune.py` не запускался, `train.py` использует гиперпараметры из `config.py`
> (уже оптимизированные: MAE ≈ 7,05 на валидации).

### Пайплайн 2 — AutoGluon

> Перед запуском рекомендуется выполнить `train.py` — он создаёт `pi_scores.json`,
> который AutoGluon использует для отсечения шумовых признаков.

```bash
# Обучение (по умолчанию 1 час, preset high_quality)
.venv_ag\Scripts\python train_autogluon.py

# Больше времени = лучше результат
.venv_ag\Scripts\python train_autogluon.py --time_limit 7200 --preset best_quality

# Инференс → predictions_autogluon.csv
.venv_ag\Scripts\python inference_autogluon.py

# Если AutoGluon дал лучший результат — заменить сабмит
copy predictions_autogluon.csv predictions.csv
```

**Параметры `train_autogluon.py`:**

| Параметр | По умолчанию | Описание |
|---|---|---|
| `--time_limit` | `3600` | Бюджет времени в секундах |
| `--preset` | `high_quality` | `medium_quality` / `high_quality` / `best_quality` |

## Признаки

`src/preprocessing.build_features` строит **~190 признаков** из 19 сырых столбцов:

| Группа | Что включает |
|---|---|
| Циклические календарные | sin/cos от часа, месяца, дня года, дня недели |
| Декодированные направления | sin/cos каждого `wind_direction_*`, (u, v)-компоненты на 4 высотах |
| Wind veer | Циклическая разность направлений между соседними высотами (индикатор стабильности) |
| Физика ветра | `v²`, `v³` на каждой высоте; вертикальные сдвиги; hub-height прокси (среднее 80m+120m) |
| Доступная мощность | Доля рабочих турбин, `v³ × available_fraction`, `ρ·v³_hub × available_fraction` |
| Термодинамика | Плотность воздуха `ρ = P/(R·T)`, `ρ·v³`, `ρ·v³_hub` |
| Параметрическая power curve | Siemens Gamesa SG 3.4-132: cubic ramp 3→13 м/с, plateau до 25 м/с, cut-out; применена к `v_80m`, `v_120m`, `v_hub`, `v_mean` |
| Дополнительные | `wind_speed_std`, `precip_total`, `icing_risk`, `turbulence_index`, `pressure_change_1h/3h` |
| Лаги | Сдвиги на −1, −2, −3, −6, −12, −24, −48 ч по ключевым метеопеременным |
| Лиды | Сдвиги на +1, +2, +3, +4, +5, +6 ч (легальны: метеоданные — это прогноз, известный заранее) |
| Rolling stats | mean и std за 3, 6, 12, 24 ч по скоростям ветра и порывам |

При инференсе к тестовым данным **спереди дописывается 48-часовой хвост** обучающей выборки — чтобы первые часы тестового периода получили корректные лаги из предыдущего месяца.

## Модели

### LightGBM-ансамбль

7 членов `LGBMRegressor`, финальное предсказание — простое среднее. Гиперпараметры подобраны Optuna (MAE на валидации):

- Все члены: objective `regression_l1`, lr ≈ 0.01427, num_leaves ≈ 114
- Разнообразие: разные random seed, небольшие вариации `num_leaves`, `reg_lambda`, `colsample_bytree`
- Early stopping по валидации (patience = 150), затем дообучение на полном датасете × 1.15

### AutoGluon TabularPredictor

Обучает **106 конфигураций** разных алгоритмов и строит взвешенный ансамбль лучших:

- LightGBM (несколько вариантов), XGBoost, CatBoost
- RandomForest, ExtraTrees
- NeuralNetFastAI, NeuralNetTorch

Ключевые настройки: `num_bag_folds=0`, `num_stack_levels=0` — отключены случайные k-fold фолды, вместо этого используется наш хронологический train/val сплит. Это исключает временну́ю утечку данных.

## Схема обучения

1. Хронологический сплит: первые **90 %** строк — train, последние **10 %** — validation (≈ 3 244 часа).
2. LightGBM: early stopping на validation → определяет `best_iteration` для каждого члена.
3. Финальная модель: каждый член переобучается на **полном датасете** на `best_iteration × 1.15` шагах.
4. AutoGluon: каждая из 106 моделей обучается на train-части, оценивается на validation-части.

## Метрики

| Метрика | Формула | Зачем |
|---|---|---|
| **MAE** | `mean(|y − ŷ|)`, МВт | основная метрика |
| **RMSE** | `sqrt(mean((y − ŷ)²))`, МВт | штрафует большие ошибки |
| **R²** | `1 − SS_res / SS_tot` | объяснённая дисперсия |
| **nMAE %** | `MAE / 90.09 × 100` | отраслевой стандарт для ВИЭ |

## Результаты на валидации

Валидация = последние 10 % обучающих данных (≈ 3 244 часа).

| Конфигурация | MAE | nMAE |
|---|---|---|
| Базовое (HistGBM, 54 признака) | 7,63 | 8,47 % |
| LightGBM, 1 модель | ~7,15 | ~7,94 % |
| **LightGBM-ансамбль × 7 (текущий)** | **7,046** | **7,82 %** |
| AutoGluon WeightedEnsemble | *в процессе* | — |

## Воспроизводимость

`SEED = 42` фиксируется через `src/utils.set_global_seed`. При одинаковых входных CSV пайплайны дают идентичный результат.
#   h a h a t o n  
 