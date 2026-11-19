"""Сбор расширенного корпуса открытых статей о сне для локальной оценки.

Зачем: 20 статей организаторов — маленькая выборка, и настройки поиска легко
подогнать под неё. Здесь собирается большой корпус того же вида (открытый
доступ, лицензия CC BY, темы образа жизни и сна), нарезанный на разделы так же,
как в наборе организаторов. В решение этот корпус не попадает: он лежит в
data/ (в .gitignore) и нужен только для проверки, что поиск и защита не
ломаются на незнакомых статьях.

Источник: Europe PMC REST API (https://europepmc.org/RestfulWebService).
Запуск: python -m scripts.build_corpus [--per-topic 8]
"""

import argparse
import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

API = "https://www.ebi.ac.uk/europepmc/webservices/rest"
OUT = Path("data/corpus")
CHUNK_CHARS = 1900
MIN_SECTIONS, MAX_SECTIONS = 10, 70

# Темы: первые — те же, что в наборе организаторов; последние — соседние и «модные»
# методы, которых там нет. Вторые нужны, чтобы проверить отказ «совет мимо базы»,
# когда в базе вдруг появляется статья по теме, и устойчивость поиска к похожим статьям.
TOPICS = {
    "caffeine": "caffeine AND sleep",
    "alcohol": "alcohol AND sleep AND (architecture OR quality)",
    "evening_exercise": "exercise AND sleep AND (evening OR timing)",
    "light": "light exposure AND sleep AND (circadian OR melatonin)",
    "screens": "(smartphone OR screen time) AND sleep",
    "temperature": "(bedroom temperature OR thermal environment) AND sleep",
    "regularity": "(sleep regularity OR social jetlag OR chronotype)",
    "diet": "(diet OR meal timing OR nutrition) AND sleep quality",
    "naps": "(nap OR napping) AND (performance OR sleep)",
    "sleep_hygiene": "sleep hygiene AND (education OR intervention)",
    "noise": "(noise OR pink noise OR white noise) AND sleep",
    "mindfulness": "(mindfulness OR meditation OR relaxation) AND sleep quality",
    "shift_work": "shift work AND sleep AND (intervention OR health)",
    "melatonin": "melatonin supplementation AND sleep",
    "supplements": "(magnesium OR tryptophan OR glycine OR zinc) AND sleep",
    "bedding": "(weighted blanket OR mattress OR pillow) AND sleep",
    "air_quality": "(ventilation OR carbon dioxide OR air quality) AND bedroom AND sleep",
    "music": "music AND sleep quality",
    "wearables": "(wearable OR actigraphy OR sleep tracker) AND sleep AND accuracy",
    "adolescents": "school start time AND sleep",
}


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "sleep-corpus-builder/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def search(query: str, limit: int) -> list[dict]:
    full = f'(TITLE:(sleep) AND ({query})) AND OPEN_ACCESS:y AND LICENSE:"cc by" AND HAS_FT:y AND PUB_TYPE:"journal article"'
    params = urllib.parse.urlencode(
        {"query": full, "format": "json", "pageSize": limit, "resultType": "core", "sort": "CITED desc"}
    )
    return json.loads(fetch(f"{API}/search?{params}")).get("resultList", {}).get("result", [])


def text_of(node: ET.Element) -> str:
    parts = []
    for child in node.iter():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag in ("table-wrap", "fig", "disp-formula", "supplementary-material"):
            continue
        if child.text and tag not in ("label",):
            parts.append(child.text)
        if child.tail:
            parts.append(child.tail)
    return re.sub(r"\s+", " ", "".join(parts)).strip()


def walk(sec: ET.Element, path: list[str], out: list[tuple[str, list[str]]]) -> None:
    title_node = sec.find("title")
    title = text_of(title_node) if title_node is not None else ""
    here = [*path, title] if title else path
    paragraphs = [text_of(p) for p in sec.findall("p")]
    paragraphs = [p for p in paragraphs if len(p) > 40]
    if paragraphs:
        out.append((" › ".join(here) or "Main text", paragraphs))
    for child in sec.findall("sec"):
        walk(child, here, out)


def chunk(blocks: list[tuple[str, list[str]]]) -> list[dict]:
    sections = []
    for title, paragraphs in blocks:
        pieces, current = [], ""
        for paragraph in paragraphs:
            while len(paragraph) > CHUNK_CHARS:  # очень длинный абзац режем по предложению
                cut = paragraph.rfind(". ", 0, CHUNK_CHARS) + 1 or CHUNK_CHARS
                if current:
                    pieces.extend([current, paragraph[:cut]])
                else:
                    pieces.append(paragraph[:cut].strip())
                current, paragraph = "", paragraph[cut:].strip()
            if current and len(current) + len(paragraph) + 1 > CHUNK_CHARS:
                pieces.append(current)
                current = paragraph
            else:
                current = (current + " " + paragraph).strip()
        if current:
            pieces.append(current)
        for i, piece in enumerate(pieces, start=1):
            name = f"{title} ({i}/{len(pieces)})" if len(pieces) > 1 else title
            sections.append({"title": name, "text": piece})
    for n, section in enumerate(sections, start=1):
        section["section_id"] = f"s{n:03d}"
    return sections


def parse(xml: bytes) -> list[dict]:
    root = ET.fromstring(xml)
    blocks: list[tuple[str, list[str]]] = []
    abstract = root.find(".//front//abstract")
    if abstract is not None:
        direct = [text_of(p) for p in abstract.findall("p")]
        if direct:
            blocks.append(("Abstract", direct))
        for sec in abstract.findall("sec"):
            walk(sec, ["Abstract"], blocks)
    body = root.find(".//body")
    if body is not None:
        direct = [text_of(p) for p in body.findall("p") if len(text_of(p)) > 40]
        if direct:
            blocks.append(("Main text", direct))
        for sec in body.findall("sec"):
            walk(sec, [], blocks)
    return chunk(blocks)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-topic", type=int, default=8)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    articles, seen = [], set()
    for topic, query in TOPICS.items():
        kept = 0
        for hit in search(query, limit=args.per_topic * 4):
            pmcid, doi = hit.get("pmcid"), hit.get("doi")
            if kept >= args.per_topic or not pmcid or not doi or doi in seen:
                continue
            try:
                sections = parse(fetch(f"{API}/{pmcid}/fullTextXML"))
            except Exception as error:  # нет полного текста или битый XML — берём следующую
                print(f"  пропуск {pmcid}: {type(error).__name__}")
                continue
            time.sleep(0.2)
            if not MIN_SECTIONS <= len(sections) <= MAX_SECTIONS:
                continue
            seen.add(doi)
            kept += 1
            articles.append({
                "article_id": f"x_{len(articles) + 1:03d}",
                "title": re.sub(r"\s+", " ", hit.get("title", "")).strip().rstrip("."),
                "authors": [a.strip() for a in hit.get("authorString", "").rstrip(".").split(",") if a.strip()][:12] or ["n/a"],
                "year": int(hit.get("pubYear", 0)),
                "doi": doi,
                "license": {"name": "CC BY", "version": "n/a", "url": "https://creativecommons.org/licenses/by/4.0/"},
                "sections": sections,
                "topic": topic,
                "pmcid": pmcid,
            })
        print(f"{topic}: {kept}")
    (OUT / "articles.json").write_text(json.dumps(articles, ensure_ascii=False), encoding="utf-8")
    total = sum(len(a["sections"]) for a in articles)
    print(f"статей {len(articles)}, разделов {total}, файл {OUT / 'articles.json'}")


if __name__ == "__main__":
    main()
