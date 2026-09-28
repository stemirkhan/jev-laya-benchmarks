# Воспроизведение

Этот файл описывает новый запуск. Сохраненные результаты можно читать без GPU и API-ключа: [results](../results/README.md). Публичные обертки проверены офлайн; повторный платный тест при подготовке репозитория не выполнялся.

## Установка и данные

Python 3.12+, Linux, `uv`. Из корня репозитория:

```sh
uv sync --frozen --group dev
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run --no-sync pytest -q
uv run python prepare.py
uv run python prepare_embedding.py
uv run python audit.py
uv run python retrieval.py all --run retrieval-20260927
uv run python candidate_report.py --run retrieval-20260927
```

Подготовка скачивает публичные BEIR-архивы, закрепленные токенизаторы, код формата Laya и ONNX Jina. Веса реранкеров на этом этапе не скачиваются. Проверяются опубликованные MD5 архивов и SHA-256 файлов. TREC-COVID входит только в аудит длины, а итоговые кандидаты строятся для SciFact, FiQA и NFCorpus.

`retrieval-20260927` - имя каталога, которое ожидают последующие команды по умолчанию. Запуск по этапам: `bm25`, `embed`, `dense`, `fuse`, `validate` вместо `all`. Для нового имени передавайте соответствующий `--source` следующим скриптам.

Код использует полные документы, не обрезает вход при превышении лимита и не пересчитывает неудачные оценки до получения правильного ответа. Пилотные запросы берутся из train/dev, текстовые пересечения с test исключаются.

## Локальные модели

Нужна совместимая CUDA-среда. Исторические версии зависимостей закреплены в `uv.lock`. Установка и скачивание весов:

```sh
uv sync --frozen --extra inference --group dev
uv run --extra inference python gpu_check.py
uv run --extra inference python prepare_inference.py --download
```

Пилоты запускаются последовательно на одной карте:

```sh
uv run --extra inference python rerank.py run --model laya-multi --out runs/pilot-laya-multi-001
uv run --extra inference python rerank.py run --model bge --out runs/pilot-bge-001
uv run --extra inference python rerank.py run --model qwen-0.6b --out runs/pilot-qwen-0.6b-001
uv run --extra inference python rerank.py run --model qwen-8b --out runs/pilot-qwen-8b-001
```

После проверки пилотов зафиксируйте условия. Флаг `--reviewed-pilots` означает, что результаты действительно просмотрены:

```sh
uv run --extra inference python freeze_protocol.py \
  --pilot runs/pilot-laya-multi-001 \
  --pilot runs/pilot-bge-001 \
  --pilot runs/pilot-qwen-0.6b-001 \
  --pilot runs/pilot-qwen-8b-001 \
  --config config/rerank-pilot.json \
  --reviewed-pilots \
  --out config/rerank-frozen.json

uv run --extra inference python rerank.py run \
  --split test --model laya-multi \
  --config config/rerank-pilot.json \
  --protocol config/rerank-frozen.json \
  --out runs/test-laya-multi-001 \
  --max-runtime-seconds 3600
```

Последнюю команду повторите для `bge`, `qwen-0.6b`, `qwen-8b`, каждый раз меняя `--model` и имя `--out`. Протокол проверяет код, данные, настройки, окружение и устройство. После изменения машины или условий нужны новые технические пилоты и новый протокол. Исторические fingerprints не подходят к новому запуску.

## Jev через OpenRouter

Проверка плана не вызывает API:

```sh
uv run python openrouter_run.py plan --split pilot
uv run python openrouter_run.py plan --split test
```

Для реального запуска передайте `OPENROUTER_API_KEY` через окружение своего терминала. Не записывайте значение в код, конфиг, историю команд или Git. Не публикуйте журналы нового запуска без проверки.

Следующие команды платные. `$3.50` - пример общего разрешенного потолка для пилота и test, а не гарантированная цена. Клиентский учет не заменяет лимит у провайдера. Обертка проверяет публичный тариф и закрепленную версию; при изменении условий отказывается запускаться. Незавершенный запрос с неизвестной стоимостью блокирует дальнейшие расходы.

```sh
uv run python openrouter_run.py run --split pilot \
  --out runs/pilot-jev-openrouter-001 \
  --ledger runs/openrouter-budget.jsonl \
  --budget-usd 3.50 --max-api-requests 520 \
  --max-runtime-seconds 600 --allow-paid-api
```

После проверки пилота и подготовки протокола четырех GPU-моделей:

```sh
uv run python openrouter_run.py freeze \
  --pilot runs/pilot-jev-openrouter-001 \
  --base-protocol config/rerank-frozen.json \
  --out config/rerank-openrouter-frozen.json --reviewed-pilot

uv run python openrouter_run.py run --split test \
  --out runs/test-jev-openrouter-001 \
  --ledger runs/openrouter-budget.jsonl \
  --budget-usd 3.50 --max-api-requests 50880 \
  --max-runtime-seconds 7200 --allow-paid-api \
  --protocol config/rerank-openrouter-frozen.json
```

Общий ledger сохраняется между фазами; его нельзя обнулять для продолжения расходов. Лимит времени проверяется перед очередным списком: уже начатый список завершается. Публичная обертка не перезаписывает и не возобновляет попытку автоматически, не повторяет HTTP-запросы. Native-адаптер `jev_adapter.py` сохранен отдельно; в опубликованном основном сравнении использован OpenRouter.

## Расчет метрик

По новым сохраненным прогонам, без повторного вызова моделей:

```sh
uv run python evaluate_rerank.py --split test \
  --run runs/test-laya-multi-001 \
  --run runs/test-bge-001 \
  --run runs/test-qwen-0.6b-001 \
  --run runs/test-qwen-8b-001 \
  --run runs/test-jev-openrouter-001 \
  --out runs/evaluation-all-five
```

Сравнение с `pytrec_eval` включается флагом `--cross-check`, если пакет установлен отдельно. Историческая сверка использовала `pytrec-eval-terrier 0.5.10`. Основные метрики: NDCG@10 с линейными gains, Recall@10/20, MRR@10. Bootstrap для разницы с исходным поиском: 2000 выборок, seed 20260927, 95% percentile CI. Прогрев не входит в задержку, для качества используется первый измерительный повтор.

## Отдельный throughput

После успешных пилотов и фиксации протокола именно на машине замера:

```sh
uv run --extra inference python throughput.py \
  --model laya-multi --protocol config/rerank-frozen.json \
  --out runs/throughput-laya-multi-001 --max-runtime-seconds 600
```

Аналогично для трех остальных локальных моделей. Скрипт берет подготовительную выборку, а не тестовую оценку качества. В опубликованном замере - три прохода по 480 парам после прогрева, размеры батчей не подбирались до максимума. Jev здесь не измеряется.

## Ограничения повторного запуска

Скорость зависит от GPU, CPU, драйвера, CUDA и сети. Закрепленные веса и данные не гарантируют побитового совпадения вычислений на другом оборудовании. API-провайдер может перестать обслуживать историческую версию. Не ослабляйте проверки протокола молча: при смене условий сохраняйте отдельный эксперимент.
