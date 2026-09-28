"""Token-only audit. No model loading, inference, credentials or API billing."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import gc
import gzip
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
import time

os.environ.setdefault("RAYON_NUM_THREADS", "4")
from tokenizers import Tokenizer

from prepare import ROOT, DATASETS, MODELS, LAYA_CODE_REVISION, digest, write_json

QUESTION = "Is the document relevant to the query?"
INSTRUCTION = "Given a search query, retrieve documents relevant to the query."
OPTIONS = ["false: no, the statement does not hold", "true: yes, the statement holds"]
QWEN_PREFIX = (
    '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query '
    'and the Instruct provided. Note that the answer can only be "yes" or "no".'
    '<|im_end|>\n<|im_start|>user\n'
)
QWEN_SUFFIX = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
LIMITS = {"laya-en": [512], "laya-multi": [1024, 8192], "bge": [8192],
          "qwen-0.6b": [32768], "qwen-8b": [32768]}


def quantile(values: list[int], p: float) -> float:
    if not values:
        raise ValueError("Empty sample")
    ordered = sorted(values)
    position = (len(ordered) - 1) * p
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def summary(values: list[int], limits: list[int] = ()) -> dict:
    if not values:
        return {"n": 0, "min": None, "p50": None, "p95": None, "max": None,
                "over_limits": {str(limit): {"n": 0, "pct": None} for limit in limits}}
    return {"n": len(values), "min": min(values), "p50": quantile(values, .5),
            "p95": quantile(values, .95), "max": max(values),
            "over_limits": {str(limit): {"n": sum(v > limit for v in values),
                                          "pct": 100 * sum(v > limit for v in values) / len(values)}
                            for limit in limits}}


def read_jsonl(path: Path) -> dict[str, dict]:
    rows = {}
    with path.open() as source:
        for line in source:
            row = json.loads(line)
            ident = str(row["_id"])
            if ident in rows:
                raise ValueError(f"Duplicate ID {ident} in {path}")
            rows[ident] = row
    return rows


def document_text(row: dict) -> str:
    title, text = row.get("title", ""), row["text"]
    if not isinstance(title, str) or not isinstance(text, str):
        raise ValueError("Non-string title/text")
    return title + "\n" + text if title else text


def read_dataset(name: str) -> tuple[dict, dict, dict, dict]:
    folder = ROOT / "data" / "beir" / name
    docs, queries = read_jsonl(folder / "corpus.jsonl"), read_jsonl(folder / "queries.jsonl")
    qrels, split_info, query_sets, text_sets = {}, {}, {}, {}
    for path in sorted((folder / "qrels").glob("*.tsv")):
        pairs, seen = [], set()
        with path.open() as source:
            for row in csv.DictReader(source, delimiter="\t"):
                qid, did, score = row["query-id"], row["corpus-id"], int(row["score"])
                if qid not in queries or did not in docs:
                    raise ValueError(f"Dangling qrel in {name}/{path.stem}: {qid}/{did}")
                if (qid, did) in seen:
                    raise ValueError(f"Duplicate qrel in {name}/{path.stem}: {qid}/{did}")
                seen.add((qid, did))
                pairs.append((qid, did, score))
        qrels[path.stem] = pairs
        query_sets[path.stem] = {q for q, _, _ in pairs}
        text_sets[path.stem] = {" ".join(queries[q]["text"].split()).casefold() for q in query_sets[path.stem]}
        split_info[path.stem] = {
            "queries": len(query_sets[path.stem]), "judged_pairs": len(pairs),
            "positive_pairs": sum(score > 0 for _, _, score in pairs),
            "grades": dict(sorted(Counter(score for _, _, score in pairs).items())),
        }
    overlaps = {}
    for left, right in itertools.combinations(query_sets, 2):
        overlaps[f"{left}/{right}"] = {
            "query_ids": sorted(query_sets[left] & query_sets[right]),
            "normalized_text_count": len(text_sets[left] & text_sets[right]),
        }
    bodies = [document_text(d) for d in docs.values()]
    info = {"documents": len(docs), "query_records": len(queries), "splits": split_info,
            "split_overlap": overlaps,
            "empty_texts": sum(not d["text"].strip() for d in docs.values()),
            "empty_titles": sum(not d.get("title", "").strip() for d in docs.values()),
            "duplicate_document_texts": len(bodies) - len({hashlib.sha256(x.encode()).digest() for x in bodies}),
            "document_chars": summary([len(x) for x in bodies]),
            "all_qrels_references_valid": True}
    return docs, queries, qrels, info


def special(config: dict, key: str) -> str:
    value = config[key]
    return value["content"] if isinstance(value, dict) else value


def laya_header(tok: Tokenizer, config: dict, head_max_len: int) -> tuple[list[int], dict]:
    """Mirror common.build_sequence's actual header, not a fixed reserved budget."""
    mask = special(config, "mask_token")
    mask_id = tok.token_to_id(mask)
    if mask_id is None:
        raise ValueError("Missing MASK token")
    head = tok.encode("noul question: " + QUESTION.replace(mask, " "), add_special_tokens=False).ids
    option_ids = [[mask_id] + tok.encode(" " + value.replace(mask, " "), add_special_tokens=False).ids[:48]
                  for value in OPTIONS]
    option_budget = head_max_len - sum(map(len, option_ids))
    shortened = False
    if option_budget < 16:
        per = max(4, (head_max_len - 16) // max(1, len(option_ids)))
        option_ids = [ids[:per] for ids in option_ids]
        option_budget = head_max_len - sum(map(len, option_ids))
        shortened = True
    used_head = head[:max(8, option_budget)]
    cls_id, sep_id = tok.token_to_id(special(config, "cls_token")), tok.token_to_id(special(config, "sep_token"))
    if cls_id is None or sep_id is None:
        raise ValueError("Missing CLS/SEP token")
    prefix = [cls_id] + used_head + [sep_id] + list(itertools.chain.from_iterable(option_ids)) + [sep_id]
    return prefix, {"head_max_len": head_max_len, "actual_header_with_final_sep": len(prefix) + 1,
                    "instruction_truncated": len(used_head) != len(head), "options_shortened": shortened}


class Profile:
    def __init__(self, alias: str):
        self.alias, self.limits = alias, LIMITS[alias]
        folder = ROOT / "assets" / "models" / alias
        tokenizer_folder = folder / "tokenizer" if alias.startswith("laya") else folder
        self.tok = Tokenizer.from_file(str(tokenizer_folder / "tokenizer.json"))
        self.tok.no_truncation()
        self.tok.no_padding()
        self.metadata = {"limits": self.limits, "truncation": False}
        if alias.startswith("laya"):
            config = json.loads((tokenizer_folder / "tokenizer_config.json").read_text())
            agent_config = json.loads((folder / "rl_agent_config.json").read_text())
            self.mask = special(config, "mask_token")
            self.prefix, header_info = laya_header(self.tok, config, agent_config["head_max_len"])
            self.metadata.update(header_info)
            self.metadata["state_budgets"] = {str(n): n - len(self.prefix) - 1 for n in self.limits}
            self.metadata["default_max_len"] = agent_config["max_len"]
            if header_info["instruction_truncated"] or header_info["options_shortened"]:
                raise ValueError("Provisional question does not fit Laya header")
        elif alias.startswith("qwen"):
            self.overhead = len(self.tok.encode(QWEN_PREFIX, add_special_tokens=False).ids) + len(self.tok.encode(QWEN_SUFFIX, add_special_tokens=False).ids)
            self.metadata["prefix_suffix_tokens"] = self.overhead
            self.metadata["limit_basis"] = "32K model-card context; not inferred from tokenizer.model_max_length"
        else:
            self.metadata["limit_basis"] = "8192 model-card context"

    def raw_lengths(self, texts: list[str]) -> list[int]:
        return [len(e.ids) for e in self.tok.encode_batch(texts, add_special_tokens=False)]

    def full_lengths(self, pairs: list[tuple[str, str]]) -> list[int]:
        if self.alias.startswith("laya"):
            states = [json.dumps({"query": q, "document": d}, ensure_ascii=False).replace(self.mask, " ") for q, d in pairs]
            return [v + len(self.prefix) + 1 for v in self.raw_lengths(states)]
        if self.alias.startswith("qwen"):
            bodies = [f"<Instruct>: {INSTRUCTION}\n<Query>: {q}\n<Document>: {d}" for q, d in pairs]
            return [v + self.overhead for v in self.raw_lengths(bodies)]
        return [len(e.ids) for e in self.tok.encode_batch(pairs, add_special_tokens=True)]


def chunks(rows: list, size: int = 128):
    for start in range(0, len(rows), size):
        yield rows[start:start + size]


def dump_row(output, data: dict) -> None:
    output.write(json.dumps(data, ensure_ascii=False) + "\n")


def audit_profile(name: str, profile: Profile, docs: dict, queries: dict, qrels: dict, out: Path) -> dict:
    started = time.perf_counter()
    test_qids = sorted({qid for qid, _, _ in qrels["test"]})
    qtexts = [queries[q]["text"] for q in test_qids]
    query_lengths = profile.raw_lengths(qtexts)
    empty_template_lengths = profile.full_lengths([(q, "") for q in qtexts])
    longest_index = max(range(len(test_qids)), key=lambda i: (empty_template_lengths[i], test_qids[i]))
    longest_id = test_qids[longest_index]
    longest_query = queries[longest_id]["text"]
    raw, stress, longest = [], [], []
    with gzip.open(out / f"{name}-{profile.alias}-documents.jsonl.gz", "wt") as output:
        for batch in chunks(list(docs.items())):
            texts = [document_text(row) for _, row in batch]
            doc_lens = profile.raw_lengths(texts)
            full_lens = profile.full_lengths([(longest_query, text) for text in texts])
            raw.extend(doc_lens)
            stress.extend(full_lens)
            for (did, _), dlen, flen in zip(batch, doc_lens, full_lens):
                dump_row(output, {"doc_id": did, "document_tokens": dlen,
                                  "stress_query_id": longest_id, "full_input_tokens": flen})
                longest.append((flen, did))
            if len(raw) % 25088 == 0:
                print(f"{name}/{profile.alias}: {len(raw)}/{len(docs)} documents", flush=True)
    judged, positive = [], []
    with gzip.open(out / f"{name}-{profile.alias}-judged-pairs.jsonl.gz", "wt") as output:
        for batch in chunks(qrels["test"]):
            full_lens = profile.full_lengths([(queries[qid]["text"], document_text(docs[did])) for qid, did, _ in batch])
            for (qid, did, grade), length in zip(batch, full_lens):
                judged.append(length)
                if grade > 0:
                    positive.append(length)
                dump_row(output, {"query_id": qid, "doc_id": did, "grade": grade, "full_input_tokens": length})
    result = {
        "profile": profile.metadata, "query_tokens": summary(query_lengths),
        "document_tokens": summary(raw), "stress_full_input": summary(stress, profile.limits),
        "stress_query_id": longest_id, "stress_empty_template_tokens": empty_template_lengths[longest_index],
        "judged_full_input": summary(judged, profile.limits),
        "positive_full_input": summary(positive, profile.limits),
        "longest_document_ids": [{"doc_id": did, "full_input_tokens": n} for n, did in sorted(longest, reverse=True)[:5]],
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    write_json(out / f"{name}-{profile.alias}-summary.json", result)
    print(f"done: {name}/{profile.alias}, p95 doc={result['document_tokens']['p95']:.1f}, max={max(raw)}, {result['elapsed_seconds']} s", flush=True)
    return result


def number(value: float | int | None) -> str:
    if value is None:
        return "-"
    return f"{value:,.0f}".replace(",", " ")


def exceed(result: dict, key: str, limit: int) -> str:
    item = result[key]["over_limits"][str(limit)]
    if item["pct"] is None:
        return "-"
    percent = "<0.01%" if 0 < item["pct"] < .01 else f"{item['pct']:.2f}%"
    return f"{number(item['n'])} ({percent})"


def render_report(results: dict, out: Path) -> None:
    lines = ["# Проверка данных перед сравнением Jev и Laya", "",
             f"Дата UTC: {results['created_at']}. Это проверка длины, не результаты работы моделей.", "",
             "Скачаны официальные архивы BEIR; MD5 сверены с таблицей BEIR, SHA-256 сохранены. "
             "Полные веса не загружались, GPU и API моделей не использовались.", "",
             "## Наборы и длина документов", "",
             "Число токенов в этой таблице посчитано токенизатором Qwen3-Reranker-8B. "
             "Текст документа - исходные заголовок и текст, соединенные переводом строки. "
             "Медиана и P95 относятся к длине, не ко времени ответа.", "",
             "| Набор | Документов | Test-запросов | Медиана, токены | P95, токены | Максимум, токены |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, data in results["datasets"].items():
        st = data["profiles"]["qwen-8b"]["document_tokens"]
        lines.append(f"| {name} | {number(data['documents'])} | {data['splits']['test']['queries']} | {number(st['p50'])} | {number(st['p95'])} | {number(st['max'])} |")
    lines += ["", "## Сколько полных входов превышает лимит", "",
              "Стресс-проверка: каждый документ соединен с одним длинным test-запросом данного набора. "
              "Для каждого токенизатора выбран запрос с наибольшей длиной шаблона без документа. "
              "Это не выдача BM25/RRF и не точная доля превышений среди будущих top-20. "
              "Проверены весь документ, запрос, инструкции, варианты и служебные токены; текст не обрезался.", "",
              "| Набор | Laya English, 512 | Laya Multi, 1024 | Laya Multi, 8192 | BGE, 8192 | Qwen 0.6B, 32768 | Qwen 8B, 32768 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    columns = [("laya-en", 512), ("laya-multi", 1024), ("laya-multi", 8192), ("bge", 8192), ("qwen-0.6b", 32768), ("qwen-8b", 32768)]
    for name, data in results["datasets"].items():
        values = [exceed(data["profiles"][alias], "stress_full_input", limit) for alias, limit in columns]
        lines.append("| " + " | ".join([name] + values) + " |")
    lines += ["", "## Проверка реальных размеченных пар", "",
              "Здесь исходные test-пары query/document с оценкой релевантности больше нуля. "
              "Числа показывают превышение длины, а не ошибки модели. Документные оценки не переносились на обрезки.", "",
              "| Набор | Релевантных пар | Laya English, 512 | Laya Multi, 8192 | BGE, 8192 | Qwen 8B, 32768 |",
              "|---|---:|---:|---:|---:|---:|"]
    for name, data in results["datasets"].items():
        values = [exceed(data["profiles"][alias], "positive_full_input", limit)
                  for alias, limit in [("laya-en", 512), ("laya-multi", 8192), ("bge", 8192), ("qwen-8b", 32768)]]
        lines.append("| " + " | ".join([name, number(data['splits']['test']['positive_pairs'])] + values) + " |")
    lines += ["", "## Разделения и целостность", "",
              "| Набор | Train-запросов | Dev-запросов | Test-запросов | Пустой text | Повторы текста документов |",
              "|---|---:|---:|---:|---:|---:|"]
    overlap_notes = []
    for name, data in results["datasets"].items():
        counts = [str(data["splits"].get(s, {}).get("queries", "-")) for s in ("train", "dev", "test")]
        lines.append("| " + " | ".join([name] + counts + [str(data["empty_texts"]), str(data["duplicate_document_texts"])]) + " |")
        for pair, overlap in data["split_overlap"].items():
            if overlap["query_ids"] or overlap["normalized_text_count"]:
                overlap_notes.append(f"- {name}, {pair}: {len(overlap['query_ids'])} общих ID запросов, {overlap['normalized_text_count']} совпадений нормализованного текста.")
    if overlap_notes:
        lines += ["", "Пересечения между разделениями (текст приведен к одному регистру и нормализованы пробелы):", ""] + overlap_notes
        lines += ["", "При подготовке пилота исключить совпадения из train/dev, не из test. Сами официальные файлы не изменять."]
    lines += ["", "Все ссылки qrels на запросы и документы проверены. Дубликаты ID или пар qrels вызывают ошибку проверки. "
              "Официальные разделения не изменялись; TREC-COVID не содержит отдельного train/dev в этом архиве. "
              "Пустой text не обязательно означает пустой документ: заголовок сохраняется. Повторы текста не удалялись.", "",
              "## Как посчитан вход", "",
              "- Laya: `state = {query, document}`, JSON с `ensure_ascii=False`, Noul с вопросом "
              f'`{QUESTION}` и стандартными вариантами true/false. Заголовок вопроса воспроизводит `build_sequence` из зафиксированного исходного кода.',
              "- BGE: штатная парная токенизация query/document со специальными токенами.",
              "- Qwen: официальный шаблон Instruct/Query/Document с system-prefix и assistant-suffix; обрезка отключена.",
              "- Инструкции пока предварительные. После их выбора и построения BM25/RRF нужно повторить проверку на фактических кандидатах.",
              "- Токены Jev здесь не измерены: локальный счетчик для его фактического API-входа не подключался. "
              "Окно другой модели не используется как доказательство, что вход поместится у Jev.", ""]
    first = next(iter(results["datasets"].values()))
    for alias in ("laya-en", "laya-multi"):
        meta = first["profiles"][alias]["profile"]
        budgets = ", ".join(f"{limit}: {room}" for limit, room in meta["state_budgets"].items())
        lines.append(f"{alias}: фактический заголовок с конечным SEP - {meta['actual_header_with_final_sep']} токенов; "
                     f"лимит head_max_len - {meta['head_max_len']}; место под state (полный лимит: остаток) - {budgets}.")
    lines += ["", "`head_max_len` ограничивает заголовок, но не резервирует весь этот объем для короткого вопроса. "
              "Поэтому вычитание фиксированных 192 токенов из 512 было бы слишком грубой оценкой.", "",
              "## Воспроизведение и источники", "",
              "```sh", "uv sync --frozen --group dev", "uv run python prepare.py", "uv run pytest -q", "uv run python audit.py", "```", "",
              "Точные ревизии токенизаторов, хеши архивов, версии окружения и форматы входа находятся в `summary.json` и `manifests/`. "
              "Длины каждого документа и размеченной пары сохранены в `.jsonl.gz`; содержимое исходных архивов не менялось.", "",
              "- [BEIR: архивы, контрольные суммы и условия использования](https://github.com/beir-cellar/beir#beers-available-datasets).",
              f"- [Laya: фактическая сборка входа](https://github.com/NandhaKishorM/laya/blob/{LAYA_CODE_REVISION}/laya/common.py#L94-L146).",
              "- [BGE-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3).",
              "- [Qwen3-Reranker-8B](https://huggingface.co/Qwen/Qwen3-Reranker-8B).", "",
              "Карточки наборов сохранены локально для проверки лицензий. Публикация текстов наборов этим аудитом не разрешается; "
              "исходные данные и сырые результаты никуда не загружались.", ""]
    (out / "report.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--render-summary", type=Path, help="Regenerate report.md from an existing summary; no tokenization")
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    parser.add_argument("--run-name", default="input-audit-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    args = parser.parse_args()
    if args.render_summary:
        source = args.render_summary.resolve()
        render_report(json.loads(source.read_text()), source.parent)
        shutil.copy2(Path(__file__), source.parent / "report-renderer.py")
        print(f"Report regenerated from unchanged counts: {source.parent / 'report.md'}", flush=True)
        return
    if Path(args.run_name).name != args.run_name or args.run_name in (".", ".."):
        raise ValueError("run-name must be a single directory name")
    out = ROOT / "runs" / args.run_name
    out.mkdir(parents=True, exist_ok=False)
    manifests = {p.stem: json.loads(p.read_text()) for p in sorted((ROOT / "assets" / "manifests").glob("*.json"))}
    shutil.copytree(ROOT / "assets" / "manifests", out / "manifests")
    shutil.copytree(ROOT / "assets" / "sources", out / "sources")
    for filename in ("prepare.py", "audit.py", "pyproject.toml", "uv.lock"):
        shutil.copy2(ROOT / filename, out / filename)
    results = {"created_at": datetime.now(timezone.utc).isoformat(), "mode": "tokenizers_only_no_inference",
               "python": sys.version, "platform": platform.platform(), "rayon_threads": os.environ["RAYON_NUM_THREADS"],
               "packages": {name: importlib.metadata.version(name) for name in ("tokenizers", "requests")},
               "manifests": manifests, "laya_code_revision": LAYA_CODE_REVISION,
               "templates": {"noul_question": QUESTION, "noul_options": OPTIONS, "qwen_instruction": INSTRUCTION,
                             "qwen_prefix": QWEN_PREFIX, "qwen_suffix": QWEN_SUFFIX},
               "datasets": {}}
    for name in args.datasets:
        docs, queries, qrels, info = read_dataset(name)
        info["profiles"] = {}
        results["datasets"][name] = info
        for alias in MODELS:
            profile = Profile(alias)
            info["profiles"][alias] = audit_profile(name, profile, docs, queries, qrels, out)
            write_json(out / "summary.partial.json", results)
            del profile
            gc.collect()
        del docs, queries, qrels
        gc.collect()
    write_json(out / "summary.json", results)
    render_report(results, out)
    write_json(ROOT / "runs" / "latest.json", {"run": args.run_name})
    print(f"Completed: {out / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
