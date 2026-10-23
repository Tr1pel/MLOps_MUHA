# MLOps MUHA

Платформа для обучения, сравнения и запуска моделей управления виртуальной мухой.
Одна из будущих моделей использует небольшой подграф коннектома FlyWire.

Текущий этап — асинхронное API с PostgreSQL, тестами, Docker и публикацией образов в GHCR.
MLflow, модели, среда, RabbitMQ и воркеры относятся к следующим этапам.

## Запуск

Нужны Docker Desktop с работающим Linux Engine и Docker Compose v2.
Для запуска Python вне Docker нужны Python 3.12 и [uv](https://docs.astral.sh/uv/getting-started/installation/).
В проекте и CI используется uv 0.13.5; версии Python-пакетов зафиксированы в `uv.lock`.

```powershell
Copy-Item .env.example .env
# Укажите свой POSTGRES_PASSWORD в .env (пароль только для локальной разработки).
docker compose up --build --wait
docker compose ps
```

API: [localhost:8000](http://localhost:8000/docs). PostgreSQL: `localhost:5432`.
Порты опубликованы только на loopback. В контейнерной сети приложение обращается к `db:5432`.
Данные PostgreSQL хранятся в именованном volume и сохраняются при `docker compose down`.

```powershell
Invoke-RestMethod http://localhost:8000/healthz
Invoke-RestMethod http://localhost:8000/api/v1/version
Invoke-RestMethod http://localhost:8000/api/v1/health
docker compose logs --tail 30 app
docker compose down
```

Если порт занят, задайте `APP_PORT` или `POSTGRES_PORT` в `.env`.
После изменения пароля уже инициализированного PostgreSQL недостаточно поменять `.env`:
существующая БД сохраняет прежний пароль. Используйте исходный пароль либо измените его в БД.
Команда `docker compose down --volumes` удаляет данные; она нужна только для сброса тестовой БД.

## Разработка и тесты

```powershell
uv sync --locked --group web
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
uv run --locked --group web pytest --cov=agent_platform --cov-report=term-missing --cov-report=xml
```

Без `RUN_INTEGRATION=1` тест с настоящей БД пропускается. Для полного прогона:

```powershell
docker compose up -d --wait db
$env:RUN_INTEGRATION = "1"
uv run --locked --group web pytest --cov=agent_platform --cov-report=term-missing --cov-report=xml
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

`/healthz` — служебный стабильный адрес для инфраструктуры, поэтому он вынесен
из версионированного пользовательского API. Docker healthcheck использует его;
для проверки готовности обслуживать запросы с БД используйте `/api/v1/health`.
`depends_on: service_healthy` обеспечивает начальный порядок запуска, а не постоянный мониторинг БД.

В `/api/v1/health` SQL-запрос `SELECT current_setting('server_version')` получает
**версию сервера PostgreSQL**, а не драйвера. Время `response_time_ms` измеряется
через `perf_counter` и включает получение соединения из пула, подключение, запрос
и возврат соединения. `HEALTH_TIMEOUT_SECONDS` (по умолчанию 2 секунды, максимум 30)
ограничивает эту операцию. При ошибке возвращаются `version: null` и категория
`unavailable` либо `timeout`. Внешних компонентов на текущем этапе один — PostgreSQL.

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

## Устройство проекта

```text
src/agent_platform/
  main.py             # Фабрика FastAPI, lifespan, создание и dispose движка
  api/                # Маршруты и модели HTTP-ответов
  core/               # Настройки, версия пакета, JSON-логи и request ID
  db/                 # Async SQLAlchemy + asyncpg и запрос версии сервера
tests/                # Unit-тесты и integration/ с настоящим PostgreSQL
scripts/              # Smoke-проверка API и проверка тегов публикации
.github/workflows/    # CI, Compose-проверка отказа БД и публикация в GHCR
```

Движок создаётся в lifespan без обязательного подключения и освобождается при остановке.
Все обработчики и сетевые обращения к БД асинхронны. Блокирующих SQL-драйверов,
`time.sleep` и вычислительных ML-задач в event loop нет.

Приложение пишет JSON в stdout: UTC timestamp, уровень, событие, request ID,
метод, путь, HTTP-статус и время. Ответ содержит `X-Request-ID`.
На старте пишутся версия и SHA; при сбое БД — тип исключения и длительность.
Строка подключения, текст исключения драйвера, query string и тела запросов не логируются.
В Compose включена ротация логов. Uvicorn access log отключён, чтобы не дублировать запросы.

Dockerfile использует два этапа, слой зависимостей из lock-файла и установку пакета
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
