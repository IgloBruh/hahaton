# Прогноз почасовой выработки ВЭС

Решение задачи прогнозирования почасовой генерации ветроэлектростанции мощностью **90,09 МВт** (26 турбин Siemens Gamesa SG 3.4-132, координаты 46,83° с.ш. 38,72° в.д., высота ротора 80 м) на основе метеопрогноза.

## Структура проекта

```
.
├── config.py                  # пути, гиперпараметры, ENSEMBLE_CONFIGS
├── train.py                   # обучение LightGBM-ансамбля -> models/model.pkl
├── tune.py                    # Optuna-поиск гиперпараметров LightGBM
├── inference.py               # инференс LightGBM -> predictions.csv
├── train_autogluon.py         # обучение AutoGluon -> models/autogluon_predictor/
├── inference_autogluon.py     # инференс AutoGluon -> predictions_autogluon.csv
├── train_lstm.py              # обучение LSTM -> models/lstm_model.pt
├── inference_lstm.py          # инференс LSTM -> predictions_lstm.csv
├── data/
│   ├── train_dataset.csv      # обучающая выборка (с целевым столбцом)
│   └── valid_features.csv     # тестовая выборка (без цели, для сабмита)
├── models/
│   ├── model.pkl              # LightGBM-ансамбль (создаётся train.py)
│   ├── autogluon_predictor/   # AutoGluon-предиктор (создаётся train_autogluon.py)
│   ├── lstm_model.pt          # LSTM-веса (создаётся train_lstm.py)
│   └── lstm_scaler.pkl        # StandardScaler для LSTM-признаков
└── src/
    ├── data_loader.py         # чтение CSV
    ├── preprocessing.py       # очистка + ~250 признаков + лаги таргета
    ├── model.py               # LightGBM-ансамбль
    └── utils.py               # сидирование, метрики
```

## Быстрый старт

Две команды — первая запускается один раз (установка), вторая запускает полный пайплайн.

### Шаг 1 — установка (один раз)

```powershell
py -3.14 -m venv .venv; .\.venv\Scripts\pip install lightgbm optuna pandas numpy scikit-learn joblib; py -3.11 -m venv .venv_ag; .\.venv_ag\Scripts\pip install "autogluon.tabular[all]" lightgbm pandas numpy scikit-learn joblib
```

### Шаг 2 — полный запуск (каждый раз при новых данных)

```powershell
.\.venv\Scripts\python tune.py --trials 200; .\.venv\Scripts\python train.py; .\.venv_ag\Scripts\python train_autogluon.py --time_limit 10800 --preset best_quality; .\.venv_ag\Scripts\python train_lstm.py --epochs 200; .\.venv\Scripts\python inference.py; .\.venv_ag\Scripts\python inference_autogluon.py; .\.venv_ag\Scripts\python inference_lstm.py
```

> **Время выполнения:** Optuna (~40 мин) + LightGBM (~10 мин) + AutoGluon (~3 часа) + LSTM (~60 мин) = ~5 часов суммарно.
> Все шаги выполняются последовательно. AutoGluon и LSTM можно запустить параллельно в двух терминалах — см. раздел «Рекомендуемый порядок запуска».

После завершения сравните MAE трёх моделей и скопируйте лучший результат:

```powershell
# Выбрать лучшую модель вручную по MAE из логов, затем:
copy predictions.csv predictions_final.csv          # LightGBM
copy predictions_autogluon.csv predictions_final.csv  # AutoGluon
copy predictions_lstm.csv predictions_final.csv       # LSTM
```

---

## Окружения

| Окружение | Python | Что использует |
|---|---|---|
| `.venv` | 3.14 | LightGBM, Optuna |
| `.venv_ag` | 3.11 | AutoGluon, LSTM (PyTorch уже включён) |

### Создание окружений

```bash
# LightGBM / Optuna
py -3.14 -m venv .venv
.venv\Scripts\pip install lightgbm optuna pandas numpy scikit-learn joblib

# AutoGluon + LSTM (PyTorch входит в состав AutoGluon)
py -3.11 -m venv .venv_ag
.venv_ag\Scripts\pip install "autogluon.tabular[all]" lightgbm pandas numpy scikit-learn joblib
```

## Запуск

### Пайплайн 1 — LightGBM-ансамбль

```bash
# Шаг 1 (опционально): подобрать гиперпараметры через Optuna
# По умолчанию 50 проб (~15 мин), можно задать больше
.venv\Scripts\python tune.py
.venv\Scripts\python tune.py --trials 200

# Шаг 2: обучить ансамбль из 7 членов (~5-10 мин)
# Генерирует models/model.pkl и pi_scores.json
.venv\Scripts\python train.py

# Шаг 3: сгенерировать predictions.csv
.venv\Scripts\python inference.py
```

> Если `tune.py` не запускался, `train.py` использует гиперпараметры из `config.py`.

### Пайплайн 2 — AutoGluon

```bash
# Обучение (по умолчанию 1 час, preset high_quality)
.venv_ag\Scripts\python train_autogluon.py

# Рекомендуемый запуск: 3 часа, best_quality
.venv_ag\Scripts\python train_autogluon.py --time_limit 10800 --preset best_quality

# Инференс -> predictions_autogluon.csv
.venv_ag\Scripts\python inference_autogluon.py

# Использовать как основной сабмит
copy predictions_autogluon.csv predictions.csv
```

**Параметры `train_autogluon.py`:**

| Параметр | По умолчанию | Варианты |
|---|---|---|
| `--time_limit` | `3600` | любое число секунд |
| `--preset` | `high_quality` | `medium_quality` / `high_quality` / `best_quality` / `extreme_quality` |

> `extreme_quality` использует GPU-конфигурации — имеет смысл только при наличии CUDA.

### Пайплайн 3 — LSTM

```bash
# Обучение (~30-60 мин на CPU, значительно быстрее на GPU)
.venv_ag\Scripts\python train_lstm.py

# Опции
.venv_ag\Scripts\python train_lstm.py --epochs 200 --hidden_size 512 --patience 20

# Инференс -> predictions_lstm.csv
.venv_ag\Scripts\python inference_lstm.py

# Использовать как основной сабмит
copy predictions_lstm.csv predictions.csv
```

**Параметры `train_lstm.py`:**

| Параметр | По умолчанию | Описание |
|---|---|---|
| `--epochs` | `100` | Максимум эпох (есть early stopping) |
| `--hidden_size` | `256` | Размер скрытого слоя LSTM |
| `--patience` | `15` | Early stopping: сколько эпох без улучшения |
| `--lr` | `0.001` | Learning rate |
| `--batch_size` | `64` | Размер батча |

### Рекомендуемый порядок запуска

```bash
# 1. Оптимизация гиперпараметров LightGBM (можно запустить параллельно с шагом 2)
.venv\Scripts\python tune.py --trials 200

# 2. Обучение LightGBM (создаёт pi_scores.json и модель)
.venv\Scripts\python train.py

# 3. AutoGluon и LSTM запускаются параллельно в разных терминалах
.venv_ag\Scripts\python train_autogluon.py --time_limit 10800 --preset best_quality
.venv_ag\Scripts\python train_lstm.py --epochs 200

# 4. Инференс всех моделей
.venv\Scripts\python inference.py
.venv_ag\Scripts\python inference_autogluon.py
.venv_ag\Scripts\python inference_lstm.py

# 5. Выбрать лучший результат по валидационному MAE -> скопировать в predictions.csv
```

## Признаки

`src/preprocessing.build_features` строит **~250 признаков** из 19 сырых столбцов:

| Группа | Что включает |
|---|---|
| Циклические календарные | sin/cos от часа, месяца, дня года, дня недели |
| Декодированные направления | sin/cos каждого `wind_direction_*`, (u, v)-компоненты на 4 высотах |
| Wind veer | Циклическая разность направлений между соседними высотами |
| Физика ветра | `v²`, `v³` на каждой высоте; вертикальные сдвиги; hub-height прокси |
| Доступная мощность | Доля рабочих турбин, `v³ × available_fraction`, `rho*v³_hub × available_fraction` |
| Термодинамика | Плотность воздуха `rho = P/(R*T)`, `rho*v³`, `rho*v³_hub` |
| Power curve | Siemens Gamesa SG 3.4-132: cubic ramp 3-13 м/с, plateau до 25 м/с |
| Дополнительные | `wind_speed_std`, `precip_total`, `icing_risk`, `turbulence_index`, `pressure_change` |
| Лаги метео | Сдвиги на -1, -2, -3, -6, -12, -24, -48 ч по ключевым переменным |
| Лиды метео | Сдвиги на +1..+6 ч (метеоданные — прогноз, известный заранее) |
| Rolling stats | mean и std за 3, 6, 12, 24 ч по скоростям ветра |
| Лаги таргета | Прошлая выработка: lag1, lag2, lag3, lag6, lag12, lag24 ч |

> При инференсе лаги таргета заполняются **авторегрессионно**: первые значения берутся из хвоста обучающей выборки, затем используются собственные предсказания модели.

## Модели

### LightGBM-ансамбль

7 членов `LGBMRegressor`, финальное предсказание — простое среднее.

- Objective: `regression_l1` (MAE-loss)
- Гиперпараметры подобраны Optuna
- Early stopping по валидации (patience = 150)
- Финальное дообучение на полном датасете × 1.15

### AutoGluon TabularPredictor

Обучает 106 конфигураций разных алгоритмов и строит взвешенный ансамбль:

- LightGBM, XGBoost, CatBoost, RandomForest, ExtraTrees
- NeuralNetFastAI, NeuralNetTorch

`num_bag_folds=0`, `num_stack_levels=0` — отключены случайные k-fold фолды, используется хронологический train/val сплит (без утечки данных).

### LSTM

2-слойный LSTM с окном 48 часов:

- Вход: последние 48 часов всех признаков (z-score нормализация)
- Выход: выработка следующего часа
- MAE-loss, Adam, ReduceLROnPlateau, early stopping

## Метрики

| Метрика | Формула | Зачем |
|---|---|---|
| **MAE** | `mean(|y - y_pred|)`, МВт | основная метрика |
| **RMSE** | `sqrt(mean((y - y_pred)^2))`, МВт | штрафует большие ошибки |
| **R²** | `1 - SS_res / SS_tot` | объяснённая дисперсия |
| **nMAE %** | `MAE / 90.09 * 100` | отраслевой стандарт для ВИЭ |

## Результаты на валидации

Валидация = последние 10 % обучающих данных (~3 244 часа).

| Конфигурация | MAE | nMAE |
|---|---|---|
| Базовое (HistGBM, 54 признака) | 7,63 | 8,47 % |
| LightGBM-ансамбль x 7 | 7,046 | 7,82 % |
| AutoGluon WeightedEnsemble | 6,832 | 7,58 % |
| LightGBM + лаги таргета | *в процессе* | — |
| LSTM | *в процессе* | — |

## Воспроизводимость

`SEED = 42` фиксируется через `src/utils.set_global_seed`. При одинаковых входных CSV пайплайны дают идентичный результат.
