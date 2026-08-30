"""Counterfactual facet attribution and FeedbackMemRec event logging."""

from __future__ import annotations

import json
import math
import re
import threading
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional

from .evidence_credit import (
    EvidenceCreditStore,
    RelationKey,
    make_relation_key,
    relation_key_to_string,
)


_SUPPORT_ID = re.compile(r"^(user|item)[\s:_-]*(\d+)$", re.IGNORECASE)


def single_positive_ndcg_at_10(rank: int) -> float:
    """Single-positive NDCG@10 for a zero-based rank."""
    if 0 <= int(rank) < 10:
        return 1.0 / math.log2(int(rank) + 2)
    return 0.0


def relation_to_dict(key: RelationKey) -> Dict:
    target_user_id, evidence_type, evidence_id = key
    return {
        "target_user_id": target_user_id,
        "evidence_type": evidence_type,
        "evidence_id": evidence_id,
    }


class FeedbackMemRecController:
    """Owns V1 attribution, approximate source-credit updates, and logs."""

    def __init__(
        self,
        credit_store: EvidenceCreditStore,
        feedback_config: Optional[Dict] = None,
        attribution_config: Optional[Dict] = None,
    ) -> None:
        self.credit_store = credit_store
        self.feedback_config = feedback_config or {}
        self.attribution_config = attribution_config or {}
        self.log_dir: Optional[Path] = None
        self._log_lock = threading.RLock()
        self.n_attribution_episodes = 0
        self.n_counterfactual_calls = 0
        self.n_claimed_relations = 0
        self.n_validated_relations = 0
        self.facet_deltas: List[float] = []
        self.n_propagation_accepted = 0
        self.n_propagation_rejected = 0
        self.n_immediate_episode_rejected = 0
        self.n_historical_credit_rejected = 0
        self.n_direct_user_writes = 0
        self.n_direct_item_writes = 0

    def configure_log_dir(self, log_dir: str | Path) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def _append_jsonl(self, filename: str, event: Dict) -> None:
        if self.log_dir is None:
            return
        path = self.log_dir / filename
        with self._log_lock:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    @staticmethod
    def _parse_support(
        target_user_id: int, raw_support
    ) -> Optional[RelationKey]:
        if not isinstance(raw_support, str):
            return None
        match = _SUPPORT_ID.match(raw_support.strip())
        if not match:
            return None
        neighbor_type, evidence_id = match.groups()
        return make_relation_key(
            target_user_id,
            f"{neighbor_type.lower()}_neighbor",
            int(evidence_id),
        )

    @staticmethod
    def _packed_relation_set(packed_relations: List[Dict]) -> set[RelationKey]:
        packed = set()
        for relation in packed_relations or []:
            try:
                packed.add(make_relation_key(
                    relation["target_user_id"],
                    relation["evidence_type"],
                    relation["evidence_id"],
                ))
            except (KeyError, TypeError, ValueError):
                continue
        return packed

    @staticmethod
    def _rank_of(target_item: int, ranked_items: List[int]) -> int:
        try:
            return ranked_items.index(target_item)
        except ValueError:
            return len(ranked_items)

    def attribute_episode(
        self,
        episode_id: str,
        step: int,
        target_user_id: int,
        target_item: int,
        facets: List[Dict],
        full_ranked_items: List[int],
        packed_relations: List[Dict],
        rerank_with_facets: Callable[[List[Dict]], List[int]],
    ) -> Dict:
        """Run per-facet leave-one-out and update each relation once."""
        full_rank = self._rank_of(target_item, full_ranked_items)
        full_reward = single_positive_ndcg_at_10(full_rank)
        packed = self._packed_relation_set(packed_relations)
        relation_deltas = defaultdict(float)
        facet_events = []

        for facet_index, facet in enumerate(facets):
            reduced_facets = facets[:facet_index] + facets[facet_index + 1:]
            counterfactual_ranked_items = rerank_with_facets(reduced_facets)
            self.n_counterfactual_calls += 1
            counterfactual_rank = self._rank_of(target_item, counterfactual_ranked_items)
            counterfactual_reward = single_positive_ndcg_at_10(counterfactual_rank)
            facet_delta = float(full_reward - counterfactual_reward)
            self.facet_deltas.append(facet_delta)

            claimed = []
            for raw_support in facet.get("supporting_neighbors", []):
                relation = self._parse_support(target_user_id, raw_support)
                if relation is not None and relation not in claimed:
                    claimed.append(relation)
            validated = [relation for relation in claimed if relation in packed]
            self.n_claimed_relations += len(claimed)
            self.n_validated_relations += len(validated)

            if validated:
                share = facet_delta / len(validated)
                for relation in validated:
                    relation_deltas[relation] += share

            facet_events.append({
                "episode_id": str(episode_id),
                "target_user_id": int(target_user_id),
                "target_item_id": int(target_item),
                "facet_index": facet_index,
                "facet_text": facet.get("facet", facet.get("text", "")),
                "facet_confidence": facet.get("confidence"),
                "full_rank": full_rank,
                "without_facet_rank": counterfactual_rank,
                "full_reward": full_reward,
                "counterfactual_reward": counterfactual_reward,
                "facet_delta": facet_delta,
                "claimed_supporting_relations": [
                    relation_to_dict(relation) for relation in claimed
                ],
                "validated_supporting_relations": [
                    relation_to_dict(relation) for relation in validated
                ],
                "relation_credit_before": {},
                "relation_credit_after": {},
            })

        before, after = self.credit_store.update_episode(
            episode_id=episode_id,
            relation_deltas=relation_deltas,
            step=step,
            learning_rate=self.feedback_config.get("learning_rate", 0.2),
            episode_delta_clip=self.feedback_config.get("episode_delta_clip", 0.5),
        )

        for event in facet_events:
            for relation_dict in event["validated_supporting_relations"]:
                key = make_relation_key(
                    relation_dict["target_user_id"],
                    relation_dict["evidence_type"],
                    relation_dict["evidence_id"],
                )
                key_string = relation_key_to_string(key)
                if key_string in before:
                    event["relation_credit_before"][key_string] = before[key_string]
                if key_string in after:
                    event["relation_credit_after"][key_string] = after[key_string]
            self._append_jsonl("facet_attribution_events.jsonl", event)

        self.n_attribution_episodes += 1
        raw_relation_deltas = [
            {
                **relation_to_dict(relation),
                "raw_episode_delta": float(raw_delta),
            }
            for relation, raw_delta in sorted(relation_deltas.items())
        ]
        return {
            "episode_id": str(episode_id),
            "full_rank": full_rank,
            "full_reward": full_reward,
            "n_facets": len(facets),
            "n_relation_updates": len(relation_deltas),
            "raw_relation_deltas": raw_relation_deltas,
            "facet_events": facet_events,
            "relation_credit_before": before,
            "relation_credit_after": after,
        }

    def log_propagation_event(self, event: Dict) -> None:
        if event.get('decision') == 'accept':
            self.n_propagation_accepted += 1
        elif event.get('decision') == 'reject':
            self.n_propagation_rejected += 1
            if event.get('reason') == 'negative_current_episode_contribution':
                self.n_immediate_episode_rejected += 1
            elif event.get('reason') == 'negative_historical_credit':
                self.n_historical_credit_rejected += 1
        self._append_jsonl("propagation_events.jsonl", event)

    def log_direct_write_event(self, event: Dict) -> None:
        if event.get('entity_type') == 'user':
            self.n_direct_user_writes += 1
        elif event.get('entity_type') == 'item':
            self.n_direct_item_writes += 1
        self._append_jsonl("direct_write_events.jsonl", event)

    def log_read_event(self, event: Dict) -> None:
        self._append_jsonl("read_credit_events.jsonl", event)

    def log_neighbor_memory_read_event(self, event: Dict) -> None:
        """Persist one candidate dynamic-neighbor-memory audit event."""
        self._append_jsonl("neighbor_memory_read_events.jsonl", event)

    def log_packed_context_event(self, event: Dict) -> None:
        """Persist an episode-level Stage-R packing audit record."""
        self._append_jsonl("packed_context_events.jsonl", event)

    def log_read_after_write_event(self, event: Dict) -> None:
        """Persist a verified or rejected read-after-write hash comparison."""
        self._append_jsonl("read_after_write_events.jsonl", event)

    def get_stats(self) -> Dict:
        invalid = self.n_claimed_relations - self.n_validated_relations
        return {
            "n_attribution_episodes": self.n_attribution_episodes,
            "n_counterfactual_calls": self.n_counterfactual_calls,
            "n_claimed_relations": self.n_claimed_relations,
            "n_validated_relations": self.n_validated_relations,
            "n_invalid_relations": invalid,
            "invalid_relation_ratio": (
                invalid / self.n_claimed_relations if self.n_claimed_relations else 0.0
            ),
            "mean_facet_delta": (
                sum(self.facet_deltas) / len(self.facet_deltas)
                if self.facet_deltas else 0.0
            ),
            "min_facet_delta": min(self.facet_deltas) if self.facet_deltas else 0.0,
            "max_facet_delta": max(self.facet_deltas) if self.facet_deltas else 0.0,
            "n_propagation_accepted": self.n_propagation_accepted,
            "n_propagation_rejected": self.n_propagation_rejected,
            "n_immediate_episode_rejected": self.n_immediate_episode_rejected,
            "n_historical_credit_rejected": self.n_historical_credit_rejected,
            "n_direct_user_writes": self.n_direct_user_writes,
            "n_direct_item_writes": self.n_direct_item_writes,
        }
