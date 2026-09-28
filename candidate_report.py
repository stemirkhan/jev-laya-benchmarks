"""Render the candidate report from the checked run summary, never hand-copy counts."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from prepare import ROOT, digest


def number(value):
    return f"{value:,.0f}".replace(",", " ")


def render(run: Path, output: Path):
    info = json.loads((run / "summary.json").read_text())
    config = json.loads((run / "config.json").read_text())
    if hashlib.sha256(json.dumps(config, ensure_ascii=False, sort_keys=True).encode()).hexdigest() != info["config_sha256"]:
        raise ValueError("Configuration changed since validation")
    for rel, expected in info["artifacts_sha256"].items():
        if digest(run / rel) != expected:
            raise ValueError(f"Changed candidate artifact: {rel}")
    lines = ["# Кандидаты для сравнения Jev и Laya", "",
             f"Прогон: `{run.name}`. Подготовлены BM25, векторный поиск и RRF. "
             "Реранкеры еще не запускались; GPU не арендовалась, платных API-вызовов не было.", "",
             "## Зафиксированные настройки", "",
             "- Наборы: SciFact, FiQA и NFCorpus. Официальные test-разделения и документы сохранены целиком.",
             "- Laya: `convaiinnovations/laya-multilingual`, явно заданное окно 8192. English в итоговое сравнение не входит.",
             f"- BM25: `bm25s`, вариант Lucene, k1={config['bm25']['k1']}, b={config['bm25']['b']}; "
             "нижний регистр, английские стоп-слова и Snowball English stemming. Обработка одинакова для запросов и документов.",
             f"- Embeddings: `{config['embedding']['repo']}`, revision `{config['embedding']['revision']}`. "
             "Официальный ONNX, CPU, FP32, усреднение токенов с учетом маски, L2-нормализация; 512 измерений, окно 8192. Префиксов нет, обрезка отключена.",
             "- Векторный поиск: точный cosine search по всем документам набора, без приближенного индекса.",
             "- Глубина: top-100 BM25 и top-100 векторного поиска. RRF с равными весами, k=60, позиции начинаются с 1.",
             "- Итог: два списка по 20 документов на запрос - исходный BM25 и гибридный RRF. Равные оценки разрешаются по ID документа.", "",
             "Jina здесь строит поисковые векторы, а не добавляется в сравнение реранкеров. "
             "Каждый реранкер получит одинаковые ID и полный текст из общего сохраненного списка.", "",
             "## Подготовленный объем", "",
             "| Набор | Документов | Test-запросов | Пар на один вариант | Уникальных пар в двух вариантах | Среднее пересечение top-20 |",
             "|---|---:|---:|---:|---:|---:|"]
    total_queries = total_pairs = total_unique = total_pilot = 0
    for name, data in info["datasets"].items():
        test = data["splits"]["test"]
        # Sum of intersections = occurrences in both methods minus the union size.
        mean_overlap = (test["pairs_per_method"] * 2 - test["unique_pairs_both_methods"]) / test["queries"]
        lines.append(f"| {name} | {number(data['documents'])} | {number(test['queries'])} | {number(test['pairs_per_method'])} | "
                     f"{number(test['unique_pairs_both_methods'])} | {mean_overlap:.2f} из 20 |")
        total_queries += test["queries"]
        total_pairs += test["pairs_per_method"]
        total_unique += test["unique_pairs_both_methods"]
        total_pilot += data["splits"]["pilot"]["unique_pairs_both_methods"]
    lines += ["", f"Всего {number(total_queries)} test-запросов, {number(total_pairs * 2)} вхождений пар в двух вариантах "
              f"и {number(total_unique)} уникальных пар. Повторно использовать оценку совпавшей независимой пары можно "
              "для качества; замеры задержки целого списка все равно выполнять без кеша оценок.", "",
              "## Отдельная подготовительная выборка", "",
              "По четыре запроса на набор, всего 12. SciFact использует train, FiQA и NFCorpus - dev. "
              "Порядок выбран по SHA-256 от фиксированного seed и ID, без отбора по ответам моделей или оценкам релевантности.", "",
              "Тексты, совпадающие с test после нормализации регистра и пробелов, исключены только из пула для пилота. "
              "Повторы внутри этого пула тоже исключены; test не менялся. Списки ID и причины исключения сохранены в `queries.json`.", "",
              f"Для пилота готовы 480 вхождений пар в двух вариантах, {number(total_pilot)} уникальных пар. "
              "Платный запуск пилота требует отдельного согласования бюджета.", "",
              "## Проверка полного входа", "",
              "На фактических кандидатах проверены исходные документы, запросы и служебные части "
              "предварительных шаблонов Laya, BGE и обеих Qwen. Превышений выбранных окон нет. "
              "После изменения инструкции эту проверку нужно повторить. Длины в таблице указаны в токенах.", "",
              "| Набор | Максимум для embeddings документа | Максимум полного входа Laya на test | Превышений Laya 8192 |",
              "|---|---:|---:|---:|"]
    for name, data in info["datasets"].items():
        full = data["splits"]["test"]["full_input_lengths"]["laya-multi"]
        lines.append(f"| {name} | {number(data['embedding_lengths']['document-vectors']['max'])} | {number(full['max'])} | {full['over_limits']['8192']['n']} |")
    lines += ["", "Точная длина API-входа Jev пока не проверена. Числа других токенизаторов не заменяют эту проверку.", "",
              "## Запросы без совпадений в BM25", ""]
    for name, data in info["datasets"].items():
        no_match = data["splits"]["test"]["bm25_no_positive_score_queries"]
        if no_match:
            lines.append(f"В {name} у {len(no_match)} test-запросов нет лексических совпадений после токенизации и stemming. "
                         "Все оценки BM25 для них нулевые. По зафиксированному правилу равенства список заполняется по ID документа; "
                         "этот порядок не отражает релевантность. Нулевая часть исходных top-100 участвует в RRF на общих условиях. "
                         "Эти запросы не удалялись; ID сохранены в summary.json. Это нужно учитывать при разборе качества исходного поиска.")
    lines += ["",
              "## Где лежат материалы", "",
              f"Код: `{ROOT}`. Прогон: `{run}`.", "",
              "Внутри каждой папки набора:", "",
              "- `bm25.*.top100.jsonl` и `dense.*.top100.jsonl` - исходные позиции и оценки.",
              "- `rrf.*.union.jsonl` - объединенная выдача с позициями в обоих источниках.",
              "- `candidates.*.bm25.jsonl` и `candidates.*.rrf.jsonl` - общие top-20 для всех реранкеров.",
              "- `pairs.*.jsonl.gz` - уникальные пары с полными текстами и SHA-256.",
              "- `*.npy`, `*.ids.json` и `*.progress.json` - векторы, порядок ID и прогресс расчета.", "",
              "Сводка - `summary.json`; настройки - `config.json`; версии окружения - `environment.json`. "
              "Состав списков не выбирался по test-метрикам. Качество и скорость реранкеров на этом этапе не измерялись.", "",
              "## Повторный запуск", "", "```sh", "uv sync --frozen --group dev", "uv run python prepare.py",
              "uv run python prepare_embedding.py", "uv run pytest -q",
              "nice -n 10 uv run python retrieval.py all --run retrieval-repeat", "```", "",
              "При остановке повторить команду с тем же именем прогона: готовые векторы не пересчитываются. "
              "При изменении настроек нужен новый прогон. Контрольные суммы исходных файлов проверяются перед запуском.", "",
              "Источники: [BM25S](https://github.com/xhluca/bm25s), "
              "[Jina Embeddings v2 Small](https://huggingface.co/jinaai/jina-embeddings-v2-small-en). "
              "Выбор embeddings основан на размере модели и поддержке полного входа, а не на тестовом качестве.", ""]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines))
    saved_renderer = run / "candidate-report-renderer.py"
    if Path(__file__).resolve() != saved_renderer.resolve():
        shutil.copy2(Path(__file__), saved_renderer)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    if Path(a.run).name != a.run or a.run in (".", ".."):
        raise ValueError("run must be a directory name")
    run = ROOT / "runs" / a.run
    render(run, a.output or run / "report.md")
