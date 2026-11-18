"""CLINC150 loading and intent -> team routing table.

CLINC150 (Larson et al., 2019, https://github.com/clinc/oos-eval) has 150 intents
grouped into 10 domains plus an out-of-scope ("oos") class. In this project each
domain is handled by one support team; "oos" goes to a human agent and the
"small_talk" domain is answered by rule-based (ELIZA-style) replies.
"""
from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
BASE_URL = "https://raw.githubusercontent.com/clinc/oos-eval/master/data/"
FILES = ["data_full.json", "domains.json"]

OOS = "oos"

TEAM_OF_DOMAIN = {
    "banking": "Banking & Payments Team",
    "credit_cards": "Credit Card Team",
    "kitchen_and_dining": "Kitchen & Dining Team",
    "home": "Smart Home Team",
    "auto_and_commute": "Auto & Commute Team",
    "travel": "Travel Team",
    "utility": "Utility & Info Team",
    "work": "Work & HR Team",
    "meta": "Assistant Settings Team",
    "small_talk": "Auto-reply (rule-based small talk)",
}
HUMAN_AGENT = "Human Agent"


def download(data_dir: str = DATA_DIR) -> None:
    os.makedirs(data_dir, exist_ok=True)
    for f in FILES:
        path = os.path.join(data_dir, f)
        if not os.path.exists(path):
            print(f"downloading {f} ...")
            urllib.request.urlretrieve(BASE_URL + f, path)


@dataclass
class Split:
    texts: list[str]
    labels: list[str]


@dataclass
class Clinc:
    train: Split
    val: Split
    test: Split
    intents: list[str]          # 150 in-scope intents + "oos" (last)
    domain_of_intent: dict[str, str]

    @property
    def label2id(self) -> dict[str, int]:
        return {l: i for i, l in enumerate(self.intents)}

    def team(self, intent: str) -> str:
        if intent == OOS:
            return HUMAN_AGENT
        return TEAM_OF_DOMAIN[self.domain_of_intent[intent]]


def load(data_dir: str = DATA_DIR) -> Clinc:
    download(data_dir)
    with open(os.path.join(data_dir, "data_full.json")) as f:
        raw = json.load(f)
    with open(os.path.join(data_dir, "domains.json")) as f:
        domains = json.load(f)
    domain_of_intent = {i: d for d, intents in domains.items() for i in intents}

    def split(names):
        texts, labels = [], []
        for n in names:
            for t, l in raw[n]:
                texts.append(t)
                labels.append(l)
        return Split(texts, labels)

    intents = sorted(domain_of_intent) + [OOS]
    return Clinc(
        train=split(["train", "oos_train"]),
        val=split(["val", "oos_val"]),
        test=split(["test", "oos_test"]),
        intents=intents,
        domain_of_intent=domain_of_intent,
    )
