"""Request-local owner sources; selecting a source never proves its meaning."""
from __future__ import annotations

from dataclasses import dataclass
import json
import re
from types import MappingProxyType

from ..model.persona import PreferenceQuote, PREFERENCE_QUOTE_MAX_CHARACTERS
from .referenceability import REVIEWED_FACT_CONTEXT

LEGACY = "preference_quotes_v1"
SOURCE_REFS = "preference_source_refs_v1"
INDEPENDENT_RELATIONSHIP = "independent_relationship"


def source_protocol(client) -> str:
    """Only trusted client configuration selects the new wire contract."""
    selected = getattr(client, "preference_source_protocol", LEGACY)
    if selected not in (LEGACY, SOURCE_REFS):
        raise ValueError("unknown_preference_source_protocol")
    if selected == SOURCE_REFS and getattr(client, "role", None) != "memory":
        raise ValueError("preference_source_refs_require_memory_client")
    return selected


@dataclass(frozen=True)
class PreferenceSources:
    user_id: str
    session_id: str
    # The same immutable rows render the actual request's conversation. No
    # reconstructed or subsequent context is allowed to add an exposed source.
    turns: tuple[tuple[int, str, str], ...]

    def owner_text(self, index, ctx, persona) -> str | None:
        if (type(index) is not int or self.user_id != ctx.user_id
                or self.user_id != persona.user_id or self.session_id != ctx.session_id
                or not 0 <= index < len(ctx.turns)
                or index in (ctx.working_memory.get("_sensitive_memory_excluded_turns") or [])):
            return None
        text = next((user for turn, user, _ in self.turns if turn == index), None)
        if not isinstance(text, str) or not text.strip() or text != ctx.turns[index].user:
            return None
        return text

    def quote(self, index, ctx, persona) -> PreferenceQuote | None:
        text = self.owner_text(index, ctx, persona)
        if text is None or len(text) > PREFERENCE_QUOTE_MAX_CHARACTERS:
            return None
        return PreferenceQuote(session_id=self.session_id, turn_index=index, text=text)

    def fact_input(self, *, legacy_ids=False, preference_turns=()):
        """Render the complete dialogue and enumerate literal choices from the owner only."""
        catalog = {}
        rows = []
        for turn, owner, assistant in self.turns:
            choices = {}
            pieces = [owner]
            for sentence in re.findall(r".+?(?:[。！？!?；;\n]+|[.]+(?=\s|$)|\Z)", owner, re.S):
                if len(sentence) <= 240:
                    pieces.append(sentence)
                if len(sentence) > 240 or not legacy_ids:
                    pieces.extend(re.findall(r"[^，,：:]+[，,：:]*", sentence))
            for number, quote in enumerate(dict.fromkeys(pieces)):
                if 2 <= len(quote) <= 240 and quote.strip():
                    source_id = f"t{turn}" if quote == owner else f"t{turn}.s{number}"
                    if legacy_ids:
                        source_id = len(catalog)
                    choices[str(source_id)] = None if quote == owner else quote
                    catalog[source_id] = (turn, quote)
            row = {"turn_index": turn, "本人原话": owner, "助手回复": assistant, "事实出处": choices}
            if not legacy_ids and turn in preference_turns:
                row["相处说明来源"] = True
            rows.append(row)
        return (json.dumps(rows, ensure_ascii=False, separators=(",", ":")) if rows else ""), MappingProxyType(catalog)

    def bind_facts(self, catalog, episode_rows, relationship_rows, ctx, persona, *, legacy_ids=False):
        """Resolve only choices exposed in this request and still eligible after privacy review."""
        def resolve(source_id, preference_id):
            if type(source_id) not in (int, str) or source_id not in catalog:
                return None
            turn, quote = catalog[source_id]
            owner = self.owner_text(turn, ctx, persona)
            if owner is None:
                return None
            source = {"turn_index": turn, "evidence_quote": quote}
            if type(preference_id) is str and preference_id in catalog:
                control_turn, control = catalog[preference_id]
                start, other = owner.find(quote), owner.find(control)
                if (control_turn == turn and start == owner.rfind(quote) and other == owner.rfind(control)
                        and (start + len(quote) <= other or other + len(control) <= start)):
                    source["preference_source_quote"] = control
            return source

        optional = set() if legacy_ids else {"preference_source_id"}
        episodes = {}
        not_referenceable = set()

        def add_episode(key, source, keywords):
            entry = episodes.setdefault(key, {**source, "keywords": []})
            if entry.get("preference_source_quote") != source.get("preference_source_quote"):
                entry.pop("preference_source_quote", None)
            entry["keywords"].extend(word for word in keywords if word not in entry["keywords"])

        for row in episode_rows:
            if (not isinstance(row, dict) or set(row) - optional != {"source_id", "keywords", "referenceable"}
                    or type(row["referenceable"]) is not bool or not isinstance(row["keywords"], list)):
                continue
            source = resolve(row["source_id"], row.get("preference_source_id"))
            if source is None:
                continue
            if not row["referenceable"]:
                not_referenceable.add(row["source_id"])
            words = [word for word in row["keywords"] if isinstance(word, str) and word.strip()]
            matched = [word for word in words if word in source["evidence_quote"]]
            if not legacy_ids and row["keywords"]:
                if not matched:
                    continue
                if len(matched) < len(words) and any(
                    turn != source["turn_index"] and all(word in owner for word in words)
                    and self.owner_text(turn, ctx, persona) is not None
                    for turn, owner, _ in self.turns
                ):
                    continue
            add_episode(row["source_id"], source, matched)

        relationships = []
        for row in relationship_rows:
            if (not isinstance(row, dict) or set(row) - optional != {"name", "relation", "source_ids", "referenceable"}
                    or type(row["referenceable"]) is not bool or not isinstance(row["source_ids"], list)
                    or type(row["name"]) is not str or type(row["relation"]) is not str):
                continue
            name, relation = row["name"].strip(), row["relation"].strip()
            if not 1 <= len(name) <= 60 or len(relation) > 40:
                continue
            facts = []
            mixed = {}
            for source_id in row["source_ids"]:
                source = resolve(source_id, row.get("preference_source_id"))
                if source is None:
                    continue
                if source.get("preference_source_quote"):
                    mixed[source_id] = source
                    if not row["referenceable"]:
                        not_referenceable.add(source_id)
                if name not in source["evidence_quote"]:
                    continue
                if not legacy_ids and (row.get("preference_source_id") == INDEPENDENT_RELATIONSHIP
                                       or source.get("preference_source_quote")):
                    source["source_context_version"] = REVIEWED_FACT_CONTEXT
                if source not in facts:
                    facts.append(source)
            if facts and (not relation or any(relation in item["evidence_quote"] for item in facts)):
                relationships.append({"name": name, "relation": relation, "facts": facts,
                                      "referenceable": row["referenceable"]})
            else:
                facts = []
            for key, source in mixed.items():
                if source not in facts:
                    # A mixed selection already identified the independent
                    # life fact. A missing identity cannot attribute it to kin.
                    add_episode(key, source, [])
        for key, entry in episodes.items():
            entry["referenceable"] = key not in not_referenceable
        return list(episodes.values()), relationships
