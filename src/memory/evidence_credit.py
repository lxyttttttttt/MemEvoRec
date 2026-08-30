"""Target-conditional credit for collaborative evidence relations.

This store is deliberately separate from ``MemoryStorage`` and the user-item
graph. A record describes how useful one user/item neighbor has historically
been as evidence for a particular target user.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Dict, Iterable, Mapping, Tuple


RelationKey = Tuple[int, str, int]
VALID_EVIDENCE_TYPES = {"user_neighbor", "item_neighbor"}


def make_relation_key(
    target_user_id: int,
    evidence_type: str,
    evidence_id: int,
) -> RelationKey:
    if evidence_type not in VALID_EVIDENCE_TYPES:
        raise ValueError(f"Unsupported evidence_type: {evidence_type}")
    return int(target_user_id), evidence_type, int(evidence_id)


def relation_key_to_string(key: RelationKey) -> str:
    target_user_id, evidence_type, evidence_id = key
    return f"{target_user_id}|{evidence_type}|{evidence_id}"


class EvidenceCreditStore:
    """In-memory relation credit with JSON persistence and bounded updates."""

    def __init__(self) -> None:
        self._records: Dict[RelationKey, Dict] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _new_record(key: RelationKey) -> Dict:
        target_user_id, evidence_type, evidence_id = key
        return {
            "target_user_id": target_user_id,
            "evidence_type": evidence_type,
            "evidence_id": evidence_id,
            "q": 0.0,
            "num_updates": 0,
            "positive_updates": 0,
            "negative_updates": 0,
            "last_updated_step": -1,
            "last_episode_id": None,
        }

    def get(self, key: RelationKey) -> Dict:
        key = make_relation_key(*key)
        with self._lock:
            record = self._records.get(key)
            if record is None:
                record = self._new_record(key)
            return dict(record)

    def get_q(self, key: RelationKey) -> float:
        return float(self.get(key)["q"])

    def get_multiplier(
        self,
        key: RelationKey,
        lambda_credit: float = 0.5,
        min_multiplier: float = 0.5,
        max_multiplier: float = 1.5,
    ) -> float:
        multiplier = 1.0 + float(lambda_credit) * self.get_q(key)
        return float(max(min_multiplier, min(max_multiplier, multiplier)))

    def update_episode(
        self,
        episode_id: str,
        relation_deltas: Mapping[RelationKey, float],
        step: int,
        learning_rate: float = 0.2,
        episode_delta_clip: float = 0.5,
    ) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
        """Apply one aggregated update per relation for an episode.

        ``relation_deltas`` must already aggregate all facet shares belonging
        to a relation. Replaying the same episode is idempotent per relation.
        """
        before: Dict[str, Dict] = {}
        after: Dict[str, Dict] = {}
        clip = abs(float(episode_delta_clip))

        with self._lock:
            for raw_key, raw_delta in relation_deltas.items():
                key = make_relation_key(*raw_key)
                record = self._records.setdefault(key, self._new_record(key))
                key_string = relation_key_to_string(key)
                before[key_string] = dict(record)

                if record.get("last_episode_id") == episode_id:
                    after[key_string] = dict(record)
                    continue

                delta = float(max(-clip, min(clip, float(raw_delta))))
                q_new = max(-1.0, min(1.0, record["q"] + learning_rate * delta))
                record["q"] = float(q_new)
                record["num_updates"] += 1
                if delta > 0:
                    record["positive_updates"] += 1
                elif delta < 0:
                    record["negative_updates"] += 1
                record["last_updated_step"] = int(step)
                record["last_episode_id"] = str(episode_id)
                after[key_string] = dict(record)

        return before, after

    def should_accept_propagation(
        self,
        key: RelationKey,
        min_credit_observations: int = 2,
        threshold: float = -0.3,
    ) -> Tuple[bool, str, Dict]:
        record = self.get(key)
        if record["num_updates"] < int(min_credit_observations):
            return True, "insufficient_observations_explore", record
        if record["q"] >= float(threshold):
            return True, "credit_above_threshold", record
        return False, "credit_below_threshold", record

    def records(self) -> Iterable[Dict]:
        with self._lock:
            return [dict(record) for record in self._records.values()]

    def get_stats(self) -> Dict:
        records = list(self.records())
        q_values = [record["q"] for record in records]
        return {
            "n_relations": len(records),
            "n_positive": sum(q > 0 for q in q_values),
            "n_negative": sum(q < 0 for q in q_values),
            "n_neutral": sum(q == 0 for q in q_values),
            "mean_q": sum(q_values) / len(q_values) if q_values else 0.0,
            "total_updates": sum(record["num_updates"] for record in records),
        }

    def save(self, path: str | Path) -> None:
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "records": sorted(
                self.records(),
                key=lambda r: (
                    r["target_user_id"], r["evidence_type"], r["evidence_id"]
                ),
            ),
        }
        temporary_path = save_path.with_suffix(save_path.suffix + ".tmp")
        with temporary_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        temporary_path.replace(save_path)

    def load(self, path: str | Path) -> None:
        load_path = Path(path)
        with load_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        loaded: Dict[RelationKey, Dict] = {}
        for raw_record in payload.get("records", []):
            key = make_relation_key(
                raw_record["target_user_id"],
                raw_record["evidence_type"],
                raw_record["evidence_id"],
            )
            record = self._new_record(key)
            record.update(raw_record)
            record["q"] = float(max(-1.0, min(1.0, record["q"])))
            loaded[key] = record
        with self._lock:
            self._records = loaded

