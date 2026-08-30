"""
Budgeted snippet packer
Greedily select most valuable neighbor snippets under token budget constraint
"""
import hashlib
from typing import Dict, List, Optional


class SnippetPacker:
    """Greedy packer based on token budget"""
    
    def __init__(
        self,
        tau: int = 1800,
        tokenizer=None,
        neighbor_memory_config: Optional[Dict] = None,
    ):
        """
        Initialize packer
        
        Args:
            tau: Token budget (default 1800, leaving enough space for facets + candidate_notes)
        """
        self.tau = tau
        self.tokenizer = tokenizer
        self.neighbor_memory_config = neighbor_memory_config or {}
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text
        Simple heuristic: 1 token ≈ 4 characters (accurate for English)
        
        Args:
            text: Input text
            
        Returns:
            Estimated token count
        """
        return len(text) // 4

    def count_tokens(self, text: str) -> int:
        """Count exact model tokens when V1.2 supplies a tokenizer."""
        if self.tokenizer is None:
            return self.estimate_tokens(text)
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def _truncate_memory(self, memory: str, max_tokens: int) -> tuple[str, int, bool]:
        """Deterministic token-aware head-tail truncation."""
        if max_tokens <= 0:
            return "", 0, bool(memory)
        token_ids = self.tokenizer.encode(memory, add_special_tokens=False)
        if len(token_ids) <= max_tokens:
            return memory, len(token_ids), False

        marker = " … "
        marker_tokens = self.tokenizer.encode(marker, add_special_tokens=False)
        if max_tokens <= len(marker_tokens):
            return "", 0, True

        head = min(
            int(self.neighbor_memory_config.get("head_tokens", 40)),
            len(token_ids),
        )
        tail = min(
            int(self.neighbor_memory_config.get("tail_tokens", 24)),
            max(len(token_ids) - head, 0),
        )

        # The omission marker counts toward the cap. Reduce tail first, then head.
        while head + tail + len(marker_tokens) > max_tokens and tail > 0:
            tail -= 1
        while head + tail + len(marker_tokens) > max_tokens and head > 0:
            head -= 1

        def render() -> str:
            head_text = self.tokenizer.decode(
                token_ids[:head],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            tail_text = self.tokenizer.decode(
                token_ids[-tail:] if tail else [],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            return f"{head_text}{marker}{tail_text}"

        truncated = render()
        included = self.count_tokens(truncated)
        while included > max_tokens and (tail > 0 or head > 0):
            if tail > 0:
                tail -= 1
            else:
                head -= 1
            truncated = render()
            included = self.count_tokens(truncated)
        if included > max_tokens or not truncated.strip():
            return "", 0, True
        return truncated, included, True

    @staticmethod
    def _memory_hash(memory: str) -> str:
        return hashlib.sha256(memory.encode("utf-8")).hexdigest()
    
    def build_neighbor_snippet(
        self, 
        neighbor: Dict,
        dataset,
        max_len: int = 200
    ) -> str:
        """
        Build text snippet for a neighbor
        
        Args:
            neighbor: Neighbor dict {'type': 'item'/'user', 'id': int, 'score': float, ...}
            dataset: RecDataset instance (for retrieving metadata)
            max_len: Maximum snippet length (characters)
            
        Returns:
            Formatted snippet string
        """
        neighbor_type = neighbor['type']
        neighbor_id = neighbor['id']
        score = neighbor['score']
        
        if neighbor_type == 'item':
            # Item snippet: title + description (truncated)
            if dataset.item_metadata and neighbor_id in dataset.item_metadata:
                meta = dataset.item_metadata[neighbor_id]
                title = meta.get('title', f'Item-{neighbor_id}')
                desc = meta.get('description', '')
                
                # Truncate
                title = title[:80] if len(title) > 80 else title
                desc = desc[:120] if len(desc) > 120 else desc
                
                snippet = f"[Item-{neighbor_id}] {title}"
                if desc:
                    snippet += f" | {desc}"
                snippet += f" (score={score:.3f})"
            else:
                snippet = f"[Item-{neighbor_id}] (score={score:.3f})"
        
        else:  # user
            # User snippet: simple description + their popular items
            snippet = f"[User-{neighbor_id}] (overlap_score={score:.3f})"
            
            # Optional: list this user's most recent 2-3 items
            if hasattr(dataset, 'train_data') and neighbor_id in dataset.train_data:
                user_items = dataset.train_data[neighbor_id]
                recent_items = user_items[-3:] if len(user_items) >= 3 else user_items
                if recent_items and dataset.item_metadata:
                    item_titles = []
                    for iid in recent_items:
                        if iid in dataset.item_metadata:
                            title = dataset.item_metadata[iid].get('title', f'Item-{iid}')
                            item_titles.append(title[:40])
                    if item_titles:
                        snippet += f" - Recent: {', '.join(item_titles)}"
        
        return snippet[:max_len]
    
    def pack(
        self,
        pruned_subgraph: Dict,
        dataset,
        candidates: List[int] = None,
        user_memory_summary: str = "",
        neighbor_memory_snapshot: Optional[Dict] = None,
        episode_id: Optional[str] = None,
        phase: Optional[str] = None,
    ) -> Dict:
        """
        Pack neighbor snippets within token budget
        
        Args:
            pruned_subgraph: Pruned subgraph (from pruner)
            dataset: RecDataset instance
            candidates: List of candidate items
            user_memory_summary: User memory summary (from storage)
            
        Returns:
            {
                'context_text': str,  # Complete context string
                'neighbors_text': str,  # Neighbor table/list
                'candidates_text': str,  # Candidate list
                'memory_text': str,  # Memory summary
                'n_neighbors': int,
                'estimated_tokens': int
            }
        """
        neighbors = pruned_subgraph['neighbors']
        user_id = pruned_subgraph['user_id']
        
        # 1. Build neighbor snippets
        neighbor_snippets = []
        for neighbor in neighbors:
            snippet = self.build_neighbor_snippet(neighbor, dataset)
            estimated = self.estimate_tokens(snippet)
            neighbor_snippets.append({
                'text': snippet,
                'tokens': estimated,
                'score': neighbor['score'],
                'neighbor_type': neighbor['type'],
                'neighbor_id': neighbor['id'],
            })
        
        # 2. Greedy selection: sort by score, add incrementally until budget exceeded
        neighbor_snippets.sort(key=lambda x: x['score'], reverse=True)
        
        selected_snippets = []
        current_tokens = 0
        
        # Reserve space for other parts
        reserve_for_candidates = 300 if candidates else 0
        reserve_for_memory = 200 if user_memory_summary else 0
        reserve_for_output = 600  # Reserve for facets + candidate_notes output
        available_budget = self.tau - reserve_for_candidates - reserve_for_memory - reserve_for_output
        
        for snippet_info in neighbor_snippets:
            if current_tokens + snippet_info['tokens'] <= available_budget:
                selected_snippets.append(snippet_info)
                current_tokens += snippet_info['tokens']
            else:
                break
        
        # 3. Build neighbors text (table format)
        if selected_snippets:
            neighbors_text = "**Collaborative Neighbors:**\n" + "\n".join(
                f"{i+1}. {s['text']}" for i, s in enumerate(selected_snippets)
            )
        else:
            neighbors_text = "**Collaborative Neighbors:** (none available)"
        
        # 4. Build candidates text
        candidates_text = ""
        if candidates:
            cand_items = []
            for cand_id in candidates[:10]:  # Limit to 10
                if dataset.item_metadata and cand_id in dataset.item_metadata:
                    meta = dataset.item_metadata[cand_id]
                    title = meta.get('title', f'Item-{cand_id}')
                    cand_items.append(f"[{cand_id}] {title[:60]}")
                else:
                    cand_items.append(f"[{cand_id}]")
            candidates_text = "**Candidates to Rank:**\n" + "\n".join(
                f"{i+1}. {c}" for i, c in enumerate(cand_items)
            )
        
        # 5. Build memory text
        memory_text = ""
        if user_memory_summary:
            memory_text = f"**User Memory Summary:**\n{user_memory_summary[:200]}"
        
        # 6. Assemble complete context
        context_parts = []
        if memory_text:
            context_parts.append(memory_text)
        context_parts.append(neighbors_text)
        if candidates_text:
            context_parts.append(candidates_text)
        
        context_text = "\n\n".join(context_parts)
        
        # 7. Estimate total tokens
        estimated_tokens = self.estimate_tokens(context_text)

        packed_user_neighbor_ids = [
            int(s['neighbor_id']) for s in selected_snippets
            if s['neighbor_type'] == 'user'
        ]
        packed_item_neighbor_ids = [
            int(s['neighbor_id']) for s in selected_snippets
            if s['neighbor_type'] == 'item'
        ]
        packed_relations = [
            {
                'target_user_id': int(user_id),
                'evidence_type': f"{s['neighbor_type']}_neighbor",
                'evidence_id': int(s['neighbor_id']),
            }
            for s in selected_snippets
        ]
        
        baseline_result = {
            'context_text': context_text,
            'neighbors_text': neighbors_text,
            'candidates_text': candidates_text,
            'memory_text': memory_text,
            'n_neighbors': len(selected_snippets),
            'estimated_tokens': estimated_tokens,
            'packed_user_neighbor_ids': packed_user_neighbor_ids,
            'packed_item_neighbor_ids': packed_item_neighbor_ids,
            'packed_relations': packed_relations,
        }

        # Exact V1.1 path: no tokenizer use, no dynamic-memory selection, and the
        # returned prompt/context fields are byte-for-byte the baseline values.
        if not self.neighbor_memory_config.get('enabled', False):
            return baseline_result
        if self.tokenizer is None:
            raise RuntimeError(
                "neighbor_memory_read.enabled=true requires a tokenizer"
            )
        if not self.neighbor_memory_config.get('preserve_static_neighbors', True):
            raise ValueError(
                "V1.2 requires neighbor_memory_read.preserve_static_neighbors=true"
            )

        snapshot = neighbor_memory_snapshot or {}
        max_by_type = {
            'user': int(self.neighbor_memory_config.get('max_user_neighbors', 4)),
            'item': int(self.neighbor_memory_config.get('max_item_neighbors', 4)),
        }
        per_memory_cap = int(
            self.neighbor_memory_config.get('per_memory_tokens', 64)
        )
        total_memory_cap = int(
            self.neighbor_memory_config.get('total_memory_tokens', 512)
        )
        if per_memory_cap <= 0 or total_memory_cap < 0:
            raise ValueError(
                "neighbor-memory token budgets must be non-negative and "
                "per_memory_tokens must be positive"
            )

        ranked_candidates = {'user': [], 'item': []}
        for pruned_rank, neighbor in enumerate(neighbors):
            neighbor_type = neighbor['type']
            neighbor_id = int(neighbor['id'])
            memory = snapshot.get((neighbor_type, neighbor_id))
            if not memory or not str(memory).strip():
                continue
            score = float(neighbor.get(
                'selection_score',
                neighbor.get('original_selection_score', neighbor.get('score', 0.0)),
            ))
            ranked_candidates[neighbor_type].append({
                'neighbor_type': neighbor_type,
                'neighbor_id': neighbor_id,
                'rank_in_pruned_neighbors': pruned_rank,
                'score': score,
                'memory': str(memory),
            })

        dynamic_candidates = []
        for neighbor_type in ('user', 'item'):
            ranked = sorted(
                ranked_candidates[neighbor_type],
                key=lambda entry: (-entry['score'], entry['rank_in_pruned_neighbors']),
            )
            dynamic_candidates.extend(ranked[:max_by_type[neighbor_type]])

        packed_index = {
            (entry['neighbor_type'], int(entry['neighbor_id'])): index
            for index, entry in enumerate(selected_snippets)
        }
        # Allocate scarce budget to higher-scored candidates. Their rendered
        # location remains the frozen baseline packed-neighbor position.
        dynamic_candidates.sort(
            key=lambda entry: (-entry['score'], entry['rank_in_pruned_neighbors'])
        )

        static_context_tokens = self.count_tokens(context_text)
        remaining_tokens = max(self.tau - static_context_tokens, 0)
        dynamic_budget = min(total_memory_cap, remaining_tokens)
        included_blocks = {}
        memory_events = []
        dynamic_memory_tokens = 0

        def assemble(blocks: Dict) -> tuple[str, str]:
            rendered = []
            for index, snippet_info in enumerate(selected_snippets):
                static_text = snippet_info['text']
                block = blocks.get(index)
                if block:
                    static_text = (
                        f"{static_text}\nPersistent memory:\n{block}"
                    )
                rendered.append(f"{index + 1}. {static_text}")
            enhanced_neighbors = (
                "**Collaborative Neighbors:**\n" + "\n".join(rendered)
                if rendered else "**Collaborative Neighbors:** (none available)"
            )
            parts = []
            if memory_text:
                parts.append(memory_text)
            parts.append(enhanced_neighbors)
            if candidates_text:
                parts.append(candidates_text)
            return enhanced_neighbors, "\n\n".join(parts)

        for candidate in dynamic_candidates:
            neighbor_type = candidate['neighbor_type']
            neighbor_id = candidate['neighbor_id']
            memory = candidate['memory']
            original_tokens = self.count_tokens(memory)
            key = (neighbor_type, neighbor_id)
            event = {
                'episode_id': episode_id,
                'phase': phase,
                'target_user_id': int(user_id),
                'neighbor_type': f'{neighbor_type}_neighbor',
                'neighbor_id': neighbor_id,
                'rank_in_pruned_neighbors': candidate['rank_in_pruned_neighbors'],
                'score': candidate['score'],
                'memory_content_hash': self._memory_hash(memory),
                'included_memory_hash': None,
                'original_memory_tokens': original_tokens,
                'included_memory_tokens': 0,
                'truncated': original_tokens > per_memory_cap,
                'entered_packed_context': False,
                'drop_reason': None,
            }
            if key not in packed_index:
                event['drop_reason'] = 'not_in_baseline_packed_context'
                memory_events.append(event)
                continue

            available_content_tokens = min(
                per_memory_cap,
                max(dynamic_budget - dynamic_memory_tokens, 0),
            )
            if available_content_tokens <= 0:
                event['drop_reason'] = 'global_token_budget'
                memory_events.append(event)
                continue

            packed_memory, included_tokens, truncated = self._truncate_memory(
                memory, available_content_tokens
            )
            packed_slot = packed_index[key]
            while packed_memory:
                trial_blocks = {**included_blocks, packed_slot: packed_memory}
                _, trial_context = assemble(trial_blocks)
                if self.count_tokens(trial_context) <= self.tau:
                    break
                packed_memory, included_tokens, truncated = self._truncate_memory(
                    memory, included_tokens - 1
                )

            if not packed_memory:
                event['drop_reason'] = 'global_token_budget'
                event['truncated'] = True
                memory_events.append(event)
                continue

            included_blocks[packed_slot] = packed_memory
            dynamic_memory_tokens += included_tokens
            event.update({
                'included_memory_tokens': included_tokens,
                'included_memory_hash': self._memory_hash(packed_memory),
                'truncated': truncated,
                'entered_packed_context': True,
            })
            memory_events.append(event)

        enhanced_neighbors_text, enhanced_context_text = assemble(included_blocks)
        total_packed_tokens = self.count_tokens(enhanced_context_text)
        if total_packed_tokens > self.tau:
            raise AssertionError(
                f"V1.2 packed context exceeds tau: {total_packed_tokens}>{self.tau}"
            )
        if dynamic_memory_tokens > total_memory_cap:
            raise AssertionError(
                "V1.2 dynamic memory exceeds total_memory_tokens"
            )

        dynamic_ids = [
            {
                'neighbor_type': event['neighbor_type'],
                'neighbor_id': event['neighbor_id'],
            }
            for event in memory_events if event['entered_packed_context']
        ]
        return {
            **baseline_result,
            'context_text': enhanced_context_text,
            'neighbors_text': enhanced_neighbors_text,
            'estimated_tokens': total_packed_tokens,
            'baseline_context_text': context_text,
            'baseline_neighbors_text': neighbors_text,
            'baseline_static_snippets': [entry['text'] for entry in selected_snippets],
            'pruned_neighbors_count': len(neighbors),
            'pruned_neighbor_ids': [
                {
                    'neighbor_type': f"{entry['type']}_neighbor",
                    'neighbor_id': int(entry['id']),
                }
                for entry in neighbors
            ],
            'packed_neighbors_count': len(selected_snippets),
            'packed_neighbor_ids': [
                {
                    'neighbor_type': f"{entry['neighbor_type']}_neighbor",
                    'neighbor_id': int(entry['neighbor_id']),
                }
                for entry in selected_snippets
            ],
            'dynamic_memory_candidates_count': len(dynamic_candidates),
            'dynamic_memory_packed_count': len(dynamic_ids),
            'dynamic_memory_neighbor_ids': dynamic_ids,
            'static_context_tokens': static_context_tokens,
            'baseline_token_slack': remaining_tokens,
            'dynamic_memory_budget': dynamic_budget,
            'dynamic_memory_tokens': dynamic_memory_tokens,
            'total_packed_tokens': total_packed_tokens,
            'neighbor_memory_events': memory_events,
            'packed_context_hash': self._memory_hash(enhanced_context_text),
            'baseline_context_hash': self._memory_hash(context_text),
        }
