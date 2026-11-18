# MLOps MUHA

Платформа для обучения, сравнения и запуска моделей управления виртуальной мухой.
Одна из будущих моделей использует небольшой подграф коннектома FlyWire.

Текущий этап — MLflow: Tracking Server в Compose, EDA, отслеживание датасетов, сравнение
моделей, Model Registry с alias `champion` и `POST /process`, который выполняет эпизод моделью,
загруженной из MLflow при старте. RabbitMQ и воркеры относятся к следующему этапу.

## Запуск

Нужны Docker Desktop с работающим Linux Engine и Docker Compose v2.
Для запуска Python вне Docker нужны Python 3.12 и [uv](https://docs.astral.sh/uv/getting-started/installation/).
В проекте и CI используется uv 0.13.5; версии Python-пакетов зафиксированы в `uv.lock`.

```powershell
Copy-Item .env.example .env
# Укажите свои POSTGRES_PASSWORD, MLFLOW_DB_PASSWORD и MINIO_ROOT_PASSWORD в .env (только для локальной разработки).
docker compose up --build --wait
docker compose ps
```

API: [localhost:8000](http://localhost:8000/docs). MLflow UI: [localhost:5000](http://localhost:5000).
Консоль MinIO: [localhost:9001](http://localhost:9001). PostgreSQL приложения: `localhost:5432`.
Первый запуск занимает 2–3 минуты: MLflow применяет миграции к своей БД.
Порты опубликованы только на loopback. В контейнерной сети приложение обращается к `db:5432`.
Данные PostgreSQL хранятся в именованном volume и сохраняются при `docker compose down`.

```powershell
Invoke-RestMethod http://localhost:8000/healthz
Invoke-RestMethod http://localhost:8000/api/v1/version
Invoke-RestMethod http://localhost:8000/api/v1/health
docker compose logs --tail 30 app
docker compose down
```

Если порт занят, задайте `APP_PORT`, `POSTGRES_PORT`, `MLFLOW_PORT`, `MINIO_PORT` или
`MINIO_CONSOLE_PORT` в `.env`.
После изменения пароля уже инициализированного PostgreSQL недостаточно поменять `.env`:
существующая БД сохраняет прежний пароль. Используйте исходный пароль либо измените его в БД.
Команда `docker compose down --volumes` удаляет данные; она нужна только для сброса тестовой БД.

## Разработка и тесты

```powershell
uv sync --locked --group web --group train
uv run --locked pre-commit install
docker compose up -d --wait db
uv run --locked --group web uvicorn agent_platform.main:app --reload --no-access-log
```

Приложение читает `.env` и переменные окружения; окружение имеет приоритет.
Без доступной БД сервер всё равно стартует, а `/healthz` и `/api/v1/version` отвечают.
Секреты не должны попадать в Git: `.env` игнорируется, `.env.example` содержит только пример.
Пароль передаётся SQLAlchemy отдельным значением, поэтому символы `@`, `/` и `:` допустимы.

```powershell
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked pre-commit run --all-files
uv run --locked --group web --group train pytest --cov=agent_platform --cov-report=term-missing --cov-report=xml
```

Без `RUN_INTEGRATION=1` тест с настоящей БД пропускается. Для полного прогона:

```powershell
docker compose up -d --wait db
$env:RUN_INTEGRATION = "1"
uv run --locked --group web --group train pytest --cov=agent_platform --cov-report=term-missing --cov-report=xml
Remove-Item Env:RUN_INTEGRATION
```

В Bash вместо `$env:RUN_INTEGRATION = "1"` используйте `export RUN_INTEGRATION=1`.
Coverage учитывает ветвления; минимальный порог — 85%. XML-отчёт — `coverage.xml`.
Тесты проверяют успех, ошибку и таймаут БД, отмену запроса, работу liveness во время
ожидания БД, освобождение движка, версию пакета, валидацию настроек и отсутствие секретов в логах.
Интеграционный тест проходит через HTTP API, SQLAlchemy, asyncpg и настоящий PostgreSQL.

Ruff: `E4/E7/E9` и `F` — ошибки Python; `I` — порядок импортов; `UP` — современные
конструкции Python 3.12; `B` — распространённые ошибки; `ASYNC` — подозрительные блокирующие
операции в асинхронном коде. Ruff format задаёт единый стиль с длиной строки 100.
Pre-commit запускает те же версии Ruff из `uv.lock`, проверяет YAML/TOML/JSON,
конфликты, приватные ключи, регистр имён файлов, размер файлов, EOF и пробелы.
Hook устанавливается отдельно на каждой машине; CI повторяет эти проверки.

## Контракт API

| Метод и путь | Назначение | Ответ |
| --- | --- | --- |
| `GET /healthz` | Быстрая liveness-проверка процесса без обращения к БД | `200 {"status":"ok"}` |
| `GET /api/v1/version` | Идентификация установленной сборки | `200 {"version":"0.1.0","commit":"..."}` |
| `GET /api/v1/health` | End-to-end проверка PostgreSQL | `200`, либо `503` при проблеме |
| `GET /api/v1/model` | Какая модель загружена: имя, alias, версия, run_id, время загрузки | `200` |
| `POST /process` | Один эпизод виртуальной мухи загруженной моделью | `200`, `422`, `503`, `500` |

`/healthz` — служебный стабильный адрес для инфраструктуры, поэтому он вынесен
из версионированного пользовательского API. Docker healthcheck использует его;
для проверки готовности обслуживать запросы с БД используйте `/api/v1/health`.
`depends_on: service_healthy` обеспечивает начальный порядок запуска, а не постоянный мониторинг БД.

В `/api/v1/health` SQL-запрос `SELECT current_setting('server_version')` получает
**версию сервера PostgreSQL**, а не драйвера. Время `response_time_ms` измеряется
через `perf_counter` и включает получение соединения из пула, подключение, запрос
и возврат соединения. `HEALTH_TIMEOUT_SECONDS` (по умолчанию 2 секунды, максимум 30)
ограничивает эту операцию. При ошибке возвращаются `version: null` и категория
`unavailable` либо `timeout`. Эта проверка охватывает только PostgreSQL; состояние модели показывает `/api/v1/model`.

```json
{
  "status": "ok",
  "components": {
    "postgresql": {
      "status": "ok",
      "version": "17.6 (Debian ...)",
      "response_time_ms": 2.31,
      "error": null
    }
  }
}
```

**Версия API `v1`, версия приложения `0.1.0` и версия PostgreSQL — разные величины.**
Единственный источник версии приложения — `[project].version` в `pyproject.toml`.
Приложение получает её через `importlib.metadata.version("agent-platform")` из установленного
пакета, включая Docker. `APP_COMMIT` передаётся при сборке образа; локально допустим `unknown`.
Версия не задаётся независимо в обработчике, переменной окружения или имени API.

## MLflow

### Архитектура

```text
клиенты (пайплайн обучения, приложение, браузер)
        │ только HTTP
        ▼
MLflow Tracking Server (mlflow:5000)
        ├── Backend Store  → PostgreSQL `mlflow-db`: эксперименты, runs, параметры,
        │                    метрики, датасеты, Model Registry
        └── Artifact Store → MinIO, bucket `mlflow`: графики, parquet, файлы моделей
```

| Сервис | Роль |
| --- | --- |
| `mlflow` | Tracking Server и UI, образ из `mlflow/Dockerfile`, MLflow 3.18.0 |
| `mlflow-db` | Backend Store: отдельный PostgreSQL со своим volume и паролем |
| `minio`, `minio-init` | Artifact Store (S3 API) и одноразовое создание bucket |
| `pipeline` | Контейнер обучения, запускается только с профилем `train` |

Решения и их причины:

- **Отдельный контейнер `mlflow-db`**, а не вторая база в `db`: метаданные экспериментов и данные
  приложения не делят ни диск, ни пароль, ни отказ.
- **Проксирование артефактов** (`--serve-artifacts --artifacts-destination s3://mlflow`, корень
  `mlflow-artifacts:/`): клиенты ходят только по HTTP в Tracking Server, ключи MinIO есть лишь у
  сервера. Приложению и пайплайну достаточно `MLFLOW_TRACKING_URI`.
- **`--allowed-hosts`**: MLflow 3 по умолчанию отклоняет запросы с чужим заголовком `Host`.
  Разрешены `localhost`, `127.0.0.1` и имя сервиса `mlflow`; проверка не отключена.
- **`--workers 1` и `MLFLOW_SERVER_ENABLE_JOB_EXECUTION=false`**: с двумя воркерами и фоновыми
  задачами сервер не укладывался в лимит памяти 512 МБ; трекингу и реестру они не нужны.
- **Образы MinIO из `bitnamilegacy`**: официальные `minio/minio` и `minio/mc` больше не
  публикуются. Версии закреплены, но не обновляются; для продакшена нужен поддерживаемый S3.

### Данные

**Среда.** Арена 10×10, 4–7 круглых препятствий, три действия: вперёд, налево, направо. Seed
полностью определяет эпизод. Наблюдение — 9 чисел: sin и cos угла на еду относительно курса,
расстояние до еды и шесть дальномеров (−90°, −45°, −15°, +15°, +45°, +90°).

**Датасет наблюдений и действий.** Учитель на правилах проходит 800 эпизодов; пары
«наблюдение → действие» делятся **по эпизодам**: 600 в обучение (30 460 строк) и 200 в тест
(10 603 строки). Отдельные 100 seed-ов оставлены для проверки поведения в замкнутом цикле.

**Подграф FlyWire.** Источник — таблица связей FlyWire v783 из репозитория
[philshiu/Drosophila_brain_model](https://github.com/philshiu/Drosophila_brain_model),
закреплённая по SHA коммита. Берутся 400 нейронов с наибольшим числом синапсов и наибольшая сильно
связная компонента их подграфа (вошли все 400): 10 970 связей, 51,6% возбуждающих. Подграф лежит в
`data/flywire/`; URL, sha256 исходного файла и параметры отбора — в `metadata.json`.
Условия использования данных FlyWire перед публикацией нужно проверить на flywire.ai.

### Dataset tracking: source, digest, lineage

| Датасет | Где привязан | Source |
| --- | --- | --- |
| `fly-bc-train` | все runs обучения, контекст `training` | `runs:/<run генерации>/datasets/train.parquet` |
| `fly-bc-test` | все runs обучения, контекст `evaluation` | `runs:/<run генерации>/datasets/test.parquet` |
| `flywire-v783-subgraph` | runs FlyWire (контекст `connectome`) и EDA (контекст `eda`) | URL исходной таблицы с SHA коммита |

- **Source** — откуда данные взяты. Run `generate-dataset` в эксперименте `fly-data` загружает
  parquet-файлы как артефакты и записывает параметры генерации, а source датасета указывает на эти
  артефакты. Встроенного источника «артефакт run» в MLflow 3.18 нет, поэтому зарегистрирован свой
  тип `mlflow_run_artifact`.
- **Digest** — короткий отпечаток содержимого: MD5 от хешей строк pandas, числа строк и имён
  столбцов, обрезанный до 8 символов. MLflow хеширует только первые 10 000 строк, поэтому изменение
  дальше по таблице при том же числе строк digest не заметит. Это отпечаток для сравнения версий,
  а не контроль целостности.
- **Lineage** — цепочка: версия модели → run обучения → датасет (имя и digest) → source →
  run генерации и его параметры (seed-ы, версия среды, параметры учителя).

### EDA

Run `eda` сохраняет в `eda/` графики, `eda_summary.md` и `eda_summary.json`: баланс классов
действий, распределения признаков по действиям, корреляции, длины эпизодов и долю успеха учителя,
примеры траекторий, а для подграфа — распределения степеней и весов и то, как активность
распространяется по шагам. Выводы, на которые опирается обучение:

- классы несбалансированы, поэтому основная метрика — macro F1, а не accuracy;
- сигнал достигает 99% нейронов подграфа к третьему шагу, а к пятому активность выходит на
  неподвижную точку, поэтому сравниваются 2 и 6 шагов распространения.

### Эксперименты и результаты

Эксперимент `fly-behavior-cloning`, одни и те же данные и тестовые эпизоды. Основная метрика —
**macro F1 на тесте**. Вспомогательные: accuracy, log loss, precision/recall/F1 по классам, macro F1
на обучении, время предсказания и поведение в 100 отложенных эпизодах — доля успеха, столкновения,
число шагов. Учитель на этих эпизодах успешен в 96% случаев.

| Run | Модель | Macro F1 тест | Macro F1 обучение | Accuracy | Успех эпизодов | Столкновений | Шагов |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `baseline` | логистическая регрессия на наблюдениях | 0,584 | 0,603 | 0,652 | 0,76 | 18,4 | 76,6 |
| `baseline-gbdt` | градиентный бустинг на наблюдениях | **0,895** | 1,000 | 0,917 | **0,93** | 1,93 | 52,5 |
| `flywire-k2` | FlyWire, 2 шага + логистическая регрессия | 0,818 | 0,894 | 0,836 | 0,82 | 0,0 | 67,0 |
| `flywire-k6` | FlyWire, 6 шагов + логистическая регрессия | 0,831 | 0,894 | 0,845 | 0,80 | 0,0 | 70,1 |

Выводы:

- Слой FlyWire поднимает линейный классификатор с 0,58 до 0,82–0,83 macro F1: фиксированный граф
  даёт нелинейные признаки, которых не хватает логистической регрессии.
- Лучший результат у градиентного бустинга без графа. Учитель — набор пороговых правил, и деревья
  воспроизводят их напрямую; при этом на обучении F1 равен 1,0, то есть модель переобучена.
  `baseline-gbdt` добавлен как контроль: без него нельзя понять, даёт ли выигрыш структура
  коннектома или просто нелинейность.
- Разница между 2 и 6 шагами мала, как и предсказывал EDA.
- Порядок по F1 и по поведению не совпадает: FlyWire-модели не сталкиваются вовсе, но реже
  доходят до еды. Ошибки в замкнутом цикле накапливаются, поэтому эпизоды оцениваются отдельно.

Диагностические артефакты каждого run лежат в `diagnostics/`: матрица ошибок (PNG и CSV), отчёт
по классам, результаты эпизодов и траектории модели рядом с траекториями учителя.

### Model Registry

Каждый run регистрируется как версия модели `fly-policy`; теги версии содержат вариант и ключевые
метрики, версия ссылается на свой run. Alias `champion` получает версия с наибольшим macro F1 на
тесте среди тех, у кого доля успешных эпизодов не более чем на 0,10 ниже, чем у учителя.
Сейчас это версия `baseline-gbdt`. Повторный запуск пайплайна добавляет новые версии.

Модель — `sklearn.Pipeline` со всем необходимым для предсказания (для FlyWire-моделей матрица
графа и проекция входов сохранены внутри), формат сериализации — cloudpickle, поэтому приложению
нужен установленный пакет `agent_platform`.

### Запуск пайплайна

```powershell
# В контейнере, против стенда Compose:
docker compose --profile train run --rm pipeline

# На хосте:
$env:MLFLOW_TRACKING_URI = "http://localhost:5000"
uv run --locked --group web --group train python -m agent_platform.training.pipeline
uv run --locked --group web --group train python -m agent_platform.training.pipeline --stage eda train register
```

Пайплайн и `promote` читают имя модели и alias из `FLY_MODEL_NAME` и `FLY_MODEL_ALIAS`
(по умолчанию `fly-policy` и `champion`); в Compose они берутся из `MODEL_NAME` и `MODEL_ALIAS`.
Этапы: `flywire`, `generate`, `eda`, `train`, `register`. Этапам `eda` и `train` нужен
выполненный `generate` на том же сервере. `--stage flywire --rebuild-flywire` заново скачивает
исходную таблицу (100 МБ) и пересобирает подграф.

## Inference: `POST /process`

При старте приложение один раз обращается к MLflow: находит версию, на которую указывает alias
`MODEL_ALIAS` модели `MODEL_NAME`, и загружает **именно эту версию** — метаданные и веса не могут
разойтись, если alias передвинут в момент загрузки. Загрузка идёт в отдельном потоке с таймаутом
`MODEL_LOAD_TIMEOUT_SECONDS`; в лог пишутся alias, фактическая версия и run_id.
Повторных загрузок, ленивой загрузки и фонового обновления нет.

```powershell
Invoke-RestMethod http://localhost:8000/api/v1/model
Invoke-RestMethod -Method Post http://localhost:8000/process -ContentType application/json -Body '{"seed": 7, "max_steps": 200}'
uv run --locked python scripts/smoke.py --process
```

Запрос: `seed` (0…2³²−1) и необязательный `max_steps` (1…1000, по умолчанию 200). Ответ содержит
`success`, `steps`, `collisions`, `trajectory`, `food`, `obstacles` и блок `model` с именем, alias,
версией и run_id модели, выполнившей эпизод.

| Код | Когда | Тело |
| --- | --- | --- |
| `200` | эпизод выполнен | результат и блок `model` |
| `422` | некорректный запрос | стандартная ошибка валидации FastAPI |
| `503` | модель не загружена | `{"error":"model_unavailable","reason":"..."}`, где причина — `not_configured`, `alias_not_found`, `timeout` или `load_failed` |
| `503` | все слоты заняты дольше 5 секунд | `{"error":"busy"}` и заголовок `Retry-After: 1` |
| `500` | ошибка во время эпизода | `{"error":"inference_failed"}` |

Эпизод — синхронные вычисления, поэтому он выполняется в отдельном пуле потоков
(`INFERENCE_CONCURRENCY`, по умолчанию 2) и не блокирует event loop: `/healthz` отвечает, пока
идёт эпизод. Если MLflow недоступен или alias не существует, приложение всё равно стартует,
а `/process` отвечает `503`.

### Смена версии модели

```powershell
$env:MLFLOW_TRACKING_URI = "http://localhost:5000"
uv run --locked --group web --group train python -m agent_platform.training.promote              # состояние реестра
uv run --locked --group web --group train python -m agent_platform.training.promote --version 3  # champion → версия 3
docker compose restart app
Invoke-RestMethod http://localhost:8000/api/v1/model                                             # version: 3
```

Alias можно переключить и в MLflow UI. До перезапуска приложение продолжает работать со старой
версией; код inference при смене версии не меняется.

**Ограничения такого обновления.** Перезапуск единственного экземпляра — это простой, во время
которого запросы теряются. Новая версия начинает обслуживать весь трафик сразу, без проверки на
его части. Если новая версия не загрузится, приложение останется без модели и будет отвечать `503`.
Несколько экземпляров, перезапущенных в разное время, какое-то время работают на разных версиях.
Откат — это ещё одно переключение alias и ещё один перезапуск.

Как сделать лучше: держать несколько экземпляров за балансировщиком и обновлять их по одному
(rolling update) с readiness-проверкой, которая не пускает трафик, пока модель не загружена и не
прошла пробное предсказание; новую версию сначала проверять на малой доле трафика (canary) или в
теневом режиме под alias `challenger`; закреплять в деплое конкретную версию, а alias использовать
только как указатель «что выкатывать следующим», чтобы перезапуск экземпляра не менял модель
незаметно.

## Устройство проекта

```text
src/agent_platform/
  main.py             # Фабрика FastAPI, lifespan, создание и dispose движка
  api/                # Маршруты и модели HTTP-ответов
  core/               # Настройки, версия пакета, JSON-логи и request ID
  db/                 # Async SQLAlchemy + asyncpg и запрос версии сервера
  sim/                # Среда (2D-мир), учитель и запуск эпизода
  ml/                 # ConnectomeFeatures (слой FlyWire) и загрузка модели из MLflow
  training/           # Пайплайн: данные, EDA, обучение, Model Registry, promote
data/flywire/         # Подграф FlyWire v783 (400 нейронов) и его metadata.json
mlflow/               # Образ MLflow Tracking Server
tests/                # Unit-тесты и integration/ с настоящим PostgreSQL
scripts/              # Smoke-проверка API и проверка тегов публикации
.github/workflows/    # CI, Compose-проверка отказа БД и публикация в GHCR
```

Движок создаётся в lifespan без обязательного подключения и освобождается при остановке.
Все обработчики и сетевые обращения к БД асинхронны. Блокирующих SQL-драйверов,
`time.sleep` в event loop нет; эпизоды `/process` и загрузка модели выполняются в потоках.

Приложение пишет JSON в stdout: UTC timestamp, уровень, событие, request ID,
метод, путь, HTTP-статус и время. Ответ содержит `X-Request-ID`.
На старте пишутся версия и SHA; при сбое БД — тип исключения и длительность.
Строка подключения, текст исключения драйвера, query string и тела запросов не логируются.
В Compose включена ротация логов. Uvicorn access log отключён, чтобы не дублировать запросы.

Dockerfile использует отдельные этапы сборки и запуска, слой зависимостей из lock-файла и установку пакета
без editable-режима. Итоговый контейнер содержит только runtime и запускается с UID 10001.
Compose задаёт memory/CPU limits, отдельную сеть, healthchecks, volume, read-only rootfs
для приложения, сброс capabilities и `no-new-privileges`.

## CI/CD и версии образов

[GitHub Actions](https://github.com/Tr1pel/MLOps_MUHA/actions) запускается для PR,
push в `main`, тегов `v*` и вручную. Публикация выполняется только при push в `main`
или релизного тега, после успешных jobs `checks` и `container`.

`checks` проверяет lock-файл, Ruff, форматирование, pre-commit и pytest с coverage,
поднимая PostgreSQL как service container. Отчёт покрытия сохраняется как artifact.
`container` собирает Compose и проверяет три ручки, останавливает БД, проверяет
`/healthz=200` и `/api/v1/health=503`, затем запускает БД и проверяет восстановление.
Ошибка любого из этих этапов блокирует публикацию.

| Событие | Результат |
| --- | --- |
| Pull request | CI и контейнерные проверки без публикации |
| Push в `main` | `ghcr.io/tr1pel/mlops_muha:sha-<полный SHA>` |
| Push `v0.1.0` | SHA-тег и `ghcr.io/tr1pel/mlops_muha:0.1.0` |
| Ручной запуск | Только проверки |

Каждый push основной ветки даёт проверяемую сборку, связанную с точным коммитом.
Релизный тег обозначает намеренно выбранную версию. Допустим формат `vMAJOR.MINOR.PATCH`;
несовпадение тега и версии пакета останавливает публикацию. Тег `latest` не используется.
Перед новым релизом измените `[project].version`, выполните `uv lock`, проверки и коммит.

```powershell
git push origin main
git tag v0.1.0
git push origin v0.1.0
```

GHCR использует встроенный `GITHUB_TOKEN` с `packages: write` только в job публикации.
Персональный токен в secrets для workflow не нужен. Actions закреплены по SHA,
версии базовых образов и uv указаны явно. Образ получает OCI labels source, revision,
version, SBOM и provenance. После push pipeline скачивает **опубликованный образ по digest**,
запускает с PostgreSQL и проверяет API, версию и SHA. Ссылка с digest появляется в job summary.
CD в этом задании заканчивается публикацией и проверкой образа; отдельный сервер не нужен.

Для приватного репозитория пакет GHCR следует оставить приватным.
Для скачивания приватного образа с локальной машины нужна авторизация `docker login ghcr.io`
с разрешением `read:packages`; пароль вводите через стандартный безопасный вход,
не добавляйте его в файлы проекта. После входа:

```powershell
$env:APP_IMAGE = "ghcr.io/tr1pel/mlops_muha:0.1.0"
docker compose pull app
docker compose up --no-build --wait
uv run --locked python scripts/smoke.py --version 0.1.0
Remove-Item Env:APP_IMAGE
```
