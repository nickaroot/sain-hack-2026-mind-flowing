"""Сборка локального проверочного набора в формате ответа getData кейса 4.

Вопросы и атаки берутся из data/synth/heldout.json (их писала другая модель,
не та, что отвечает), статьи, эксперты и промпты проверяющей — из набора
организаторов. Опция --extra добавляет в базу посторонние статьи из
расширенного корпуса: так проверяется, что поиск не сбивается, когда база
больше и в ней есть похожие статьи.

Запуск: python -m scripts.make_dataset data/case4-dev-v2.json data/synth/heldout.json data/heldout.json [--extra 60]
"""

import argparse
import json
import random

EXPERTS = ("clinician", "biohacker", "nutritionist")
USER_FIELDS = (
    "goal", "chronotype", "weekday_bedtime", "weekday_wake_time", "average_sleep_hours", "social_jetlag",
    "activity_level", "sedentary_hours", "training_days", "works_from_home", "phone_before_sleep",
    "movement_index", "sleep_preferences", "disabled_widget_ids",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base")
    parser.add_argument("heldout")
    parser.add_argument("out")
    parser.add_argument("--extra", type=int, default=0, help="сколько посторонних статей добавить в базу")
    parser.add_argument("--corpus", default="data/corpus/articles.json")
    args = parser.parse_args()

    with open(args.base, encoding="utf-8") as f:
        base = json.load(f)["dataset"]
    with open(args.heldout, encoding="utf-8") as f:
        heldout = json.load(f)

    articles = list(base["articles"])
    if args.extra:
        with open(args.corpus, encoding="utf-8") as f:
            corpus = json.load(f)
        known = {a["doi"].lower() for a in articles}
        pool = [a for a in corpus if a["doi"].lower() not in known]
        random.Random(23).shuffle(pool)
        for article in pool[: args.extra]:
            articles.append({k: article[k] for k in ("article_id", "title", "authors", "year", "doi", "license", "sections")})
        random.Random(23).shuffle(articles)  # статьи организаторов не должны стоять первыми

    requests, reference = [], []
    for group in heldout["groups"]:
        for expert in EXPERTS:
            record_id = f"rq_{group['group_id'][2:]}_{expert[:3]}"
            requests.append({
                "record_id": record_id, "user_id": group["user_id"], "expert_id": expert,
                "group_id": group["group_id"],
                "utterance": {"role": "user", "at": group["at"], "text": group["text"]},
            })
            reference.append({
                "record_id": record_id, "request_type": group["request_type"],
                "allowed_article_ids": group["allowed_article_ids"],
                "rationale": f"[{group.get('attack_kind') or 'normal'}; {group.get('technique', '')}] {group['rationale']}",
            })

    dataset = {
        "meta": {**base["meta"], "dataset_id": "local-heldout", "phase": "dev"},
        "users": [{**dict.fromkeys(USER_FIELDS), **user} for user in heldout["users"]],
        "chat_history": heldout["chat_history"],
        "articles": articles,
        "experts": base["experts"],
        "requests": requests,
        "judge_prompts": base["judge_prompts"],
    }
    bundle = {"dataset": dataset, "reference": {"meta": {"dataset_id": "local-heldout"}, "requests": reference}}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(bundle, f, ensure_ascii=False)
    print(f"запросов {len(requests)}, троек {len(heldout['groups'])}, статей в базе {len(articles)}")


if __name__ == "__main__":
    main()
