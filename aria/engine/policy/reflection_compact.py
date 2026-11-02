"""Versioned, reversible reflection wire representation with no union grammar.

Only nullable values are represented as zero-or-one-element arrays. The
original local contract and source validator remain authoritative. This is a
transport experiment, not evidence that schema compilation caused a timeout.
"""
from __future__ import annotations

from copy import deepcopy
import json

VERSION = "reflection_json_schema_v3"
FORMAT = "json_schema_v3"
INDEXED_VERSION = "reflection_json_schema_v4"
INDEXED_FORMAT = "json_schema_v4"
_REVIEW_SLOTS = ("hypothesis_id", "claim_quote", "scope_quote")
_CHANGE_SLOTS = ("id", "scope", "lineage", "target_value")
_EXAMPLE_MARKER = "严格 JSON 形状（即使不更新也必须有protocol_version和reviews）："


def _without_annotations(value):
    if type(value) is dict:
        return {key: _without_annotations(item) for key, item in value.items()
                if key not in {"title", "description"}}
    if type(value) is list:
        return [_without_annotations(item) for item in value]
    return value


def _slot(item):
    # The provider supports minItems but not maxItems. Exact 0/1 cardinality
    # is checked locally before decoding; no second value is discarded.
    return {"type": "array", "items": deepcopy(item)}


def compact_reflection_schema():
    from .reflection_json import reflection_wire_schema
    result = _without_annotations(reflection_wire_schema())
    definitions = result["$defs"]
    for name, fields in (("_Review", _REVIEW_SLOTS), ("_Change", _CHANGE_SLOTS)):
        for field in fields:
            shape = definitions[name]["properties"][field]
            if (set(shape) != {"anyOf"} or len(shape["anyOf"]) != 2
                    or shape["anyOf"][1] != {"type": "null"}):
                raise ValueError("incompatible_compact_reflection_schema")
            definitions[name]["properties"][field] = _slot(shape["anyOf"][0])
    left = definitions.pop("_AtomicClaimNullOutcome")
    right = definitions.pop("_AtomicClaimObservedOutcome")
    expected_refs = [{"$ref": "#/$defs/_AtomicClaimNullOutcome"},
                     {"$ref": "#/$defs/_AtomicClaimObservedOutcome"}]
    atomic = definitions["_Change"]["properties"]["atomic_claim"]
    if atomic != {"anyOf": expected_refs}:
        raise ValueError("incompatible_compact_reflection_schema")
    a, b = deepcopy(left), deepcopy(right)
    for item in (a, b):
        for field in ("outcome", "evidence_basis"):
            del item["properties"][field]
    if (a != b or left["properties"]["outcome"] != {"type": "null"}
            or right["properties"]["outcome"] != {"type": "string", "pattern": r"\S"}
            or left["properties"]["evidence_basis"] != {
                "type": "string", "enum": ["stated_preference", "repeated_pattern"]}
            or right["properties"]["evidence_basis"] != {
                "type": "string", "enum": ["observed_response", "repeated_pattern"]}):
        raise ValueError("incompatible_compact_reflection_schema")
    # The cross-field basis/outcome relation is deliberately checked by the
    # unchanged original local union validator, not this simpler grammar.
    left["properties"]["outcome"] = _slot(right["properties"]["outcome"])
    left["properties"]["evidence_basis"] = {
        "type": "string", "enum": ["stated_preference", "observed_response", "repeated_pattern"]}
    definitions["_AtomicClaim"] = left
    definitions["_Change"]["properties"]["atomic_claim"] = {"$ref": "#/$defs/_AtomicClaim"}
    return result


def compact_reflection_request_format():
    return {"type": "json_schema", "json_schema": {
        "name": "aria_reflection_compact_v3", "strict": True,
        "schema": compact_reflection_schema(),
    }}


def compact_reflection_system_prompt():
    from ..memory.diagnosis import reflection_system_prompt
    original = reflection_system_prompt(True)
    if original.count(_EXAMPLE_MARKER) != 1:
        raise ValueError("incompatible_compact_reflection_prompt")
    rules, example = original.split(_EXAMPLE_MARKER, 1)
    if example.count("\n}]}\n") != 1:
        raise ValueError("incompatible_compact_reflection_prompt")
    _, suffix = example.split("\n}]}\n", 1)
    suffix = suffix.replace("lineage=null", "lineage=[]")
    return ("反思传输协议 v3。下节是领域判断规则：其中 null、字符串、对象、数字指解码后的领域值，"
            "不是你直接输出的写法。保留全部判断规则；实际输出仅遵守末尾的 v3 序列化规则。\n\n"
            + rules + "\n\nv3 序列化规则（替代旧 JSON 示例；只输出一个完整 JSON 对象）：\n"
            "顶层仅有 protocol_version、reviews、changes 三个键；protocol_version 的值固定为字符串 \"2.0\"，"
            "reviews/changes 始终是数组。传输版本 v3 和提示文档版本 2.5 都不是输出值，"
            "不得写 protocol_version=\"v3\"，不得增加 contract 字段。"
            '顶层形状示例：{"protocol_version":"2.0","reviews":[],"changes":[]}；'
            "数组内容仍按本次证据填写。所有 schema 字段都必须存在，无额外字段。不要输出解释、工具调用或思维过程。\n"
            "以下八处字段是 nullable 槽：reviews[].hypothesis_id、claim_quote、scope_quote；"
            "changes[].id、scope、lineage、target_value；changes[].atomic_claim.outcome。"
            "每个槽必须是数组且只能有零或一个元素：领域值 null 写成 []；非 null 值 x 写成 [x]。"
            "空字符串必须写成 [\"\"]，不能与 [] 混淆；有值的 lineage 写成 [{原五字段完整对象}]；"
            "数值零写成 [0]。不使用 JSON null，不把字符串 \"null\" 当缺失。\n"
            "outcome 的领域规则保持：stated_preference 写 []；observed_response 写 [\"本人有据的实际反应\"]；"
            "repeated_pattern 视原证据写 [] 或单个实际反应字符串。每槽超过一项会被拒收，不能用多项合并主张。\n"
            "其余所有字段、引文、身份、枚举、原子主张及 evidence 数组均按领域规则原样输出；"
            "不得给普通字符串额外套数组，也不得增加引用。无适用人物变化用 changes=[]。"
            "不更新仍须返回 protocol_version 和 reviews；没有任何合法 review 时 reviews=[]。\n"
            + suffix)


def compact_reflection_messages(messages):
    """Translate only the trusted system instruction; user evidence is exact."""
    from ..memory.diagnosis import reflection_system_prompt
    if (type(messages) is not list or len(messages) != 2
            or messages[0] != {"role": "system", "content": reflection_system_prompt(True)}
            or type(messages[1]) is not dict or set(messages[1]) != {"role", "content"}
            or messages[1]["role"] != "user" or type(messages[1]["content"]) is not str):
        raise ValueError("reflection_json_requires_evidence_review_contract")
    return [{"role": "system", "content": compact_reflection_system_prompt()}, deepcopy(messages[1])]


def _map_slots(value, operation):
    """No missing fields, rows, citations or defaults are created."""
    result = deepcopy(value)
    for collection, fields in (("reviews", _REVIEW_SLOTS), ("changes", _CHANGE_SLOTS)):
        rows = result.get(collection) if type(result) is dict else None
        for index, row in enumerate(rows if type(rows) is list else []):
            if type(row) is not dict:
                continue
            # JSON has no reference aliases; isolate each occurrence even
            # when a software caller reused one dictionary in several rows.
            row = deepcopy(row)
            rows[index] = row
            for field in fields:
                if field in row:
                    row[field] = operation(row[field])
            atomic = row.get("atomic_claim") if collection == "changes" else None
            if type(atomic) is dict and "outcome" in atomic:
                atomic["outcome"] = operation(atomic["outcome"])
    return result


def encode_compact_reflection(value):
    """Deterministic fixture/adapter utility; never called to repair a reply."""
    return _map_slots(value, lambda item: [] if item is None else [item])


def decode_compact_reflection(raw, *, itemwise_review=False):
    from .reflection_json import parse_reflection_json
    # This first parse rejects duplicate keys, nonfinite constants and bad
    # envelopes without touching any content. The second checks the original
    # canonical contract; its itemwise policy is unchanged.
    parsed = json.loads(parse_reflection_json(raw, itemwise_review=True))
    def decode(item):
        if type(item) is not list or len(item) > 1 or item and item[0] is None:
            raise ValueError("invalid_compact_reflection_slot")
        return item[0] if item else None
    canonical = _map_slots(parsed, decode)
    text = json.dumps(canonical, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return parse_reflection_json(text, itemwise_review=itemwise_review)


def indexed_compact_reflection_schema():
    """V4 removes model-authored bookkeeping; source/semantic fields stay exact."""
    result = compact_reflection_schema()
    review = result["$defs"]["_Review"]
    del review["properties"]["id"]
    review["required"].remove("id")
    change = result["$defs"]["_Change"]
    del change["properties"]["review_id"]
    change["required"].remove("review_id")
    change["properties"]["review_index"] = {"type": "integer", "enum": list(range(6))}
    change["required"].append("review_index")
    # JSON Schema patterns search, while some generation grammars full-match.
    # Express the same nonblank condition under either interpretation; local
    # length, itemwise and semantic validators remain authoritative.
    atomic = result["$defs"]["_AtomicClaim"]["properties"]
    text_fields = [review["properties"][key] for key in ("reason", "event_summary")]
    text_fields.extend(atomic[key] for key in ("aspect", "conditions", "tendency"))
    text_fields.append(atomic["outcome"]["items"])
    for field in text_fields:
        if field != {"type": "string", "pattern": r"\S"}:
            raise ValueError("incompatible_indexed_reflection_text_schema")
        field["pattern"] = r"[\s\S]*\S[\s\S]*"
    return result


def indexed_compact_reflection_request_format():
    return {"type": "json_schema", "json_schema": {
        "name": "aria_reflection_compact_v4", "strict": True,
        "schema": indexed_compact_reflection_schema(),
    }}


def indexed_compact_reflection_system_prompt():
    # The current route owns its instruction. No editing an old free-form
    # prompt by substring, and no second copy of the provider's strict schema.
    return """你只做会后人物证据审核，不聊天、诊断、执行动作或授予权限。
审阅本次非排除的本人原话、实际交付记录及旧假设。事实由memory整理；明确交流要求由 memory 提议并经本人确认。
偏好、分享、撤回及护理操作不自动成为人物证据。不得将助手判断或旧摘要循环当本人证据。

输出一个完整JSON对象，protocol_version 固定为字符串 "2.0"。
reviews 为数组，可为空，最多 6 项；changes最多1项。
字段、类型、枚举以本次 response_format Schema 为准，不写重复键、额外字段、Markdown或说明。
无适用证据时输出 {"protocol_version":"2.0","reviews":[],"changes":[]}，不要为填满数组写占位 review。

先审证据，再考虑更新：
1. 优先处理本人对旧主张的纠正和反证，再看本次真实互动反应。普通事实交给memory。
保留事情、主体、关系、时点及条件，不将一次情绪或年龄/性别外推为稳定人格，不把他人的
偏好算作本人证据。旧已确认偏好只读，不替本人修改或撤回；操作声明通常只作context_only。
2. evidence与event_evidence各引用1..3项本次实际输入的本人原话；turn_index沿用原编号，
evidence_quote是该轮连续逐字子串（2..240字符），不能改写、拼接、引用助手或重复原项。
evidence支持语义判断，event_evidence识别事件；纠正旧事可在后者引用本次重述，但前者
必须包含本次新纠正。reason和event_summary各一句非空说明，最多240字符，不再复述整段引文。
同一往事的改述不是新事件；事件身份、去重、计数及review的ID由代码处理。
3. applicability比较证据和原主张条件：相符within_scope，不同outside_scope，不明unknown。
新观察比较atomic_claim.conditions，不因尚无旧身份就判unknown；忙碌时短答不是空闲时的反例。
relation按support/counter/neutral/unknown判断。本人否定旧主张的一部分，记原主张的范围内counter，
不因另一部分成立改判support，不把旧claim改成相反观点。真正反证优先于支持；范围外、未知或
没试过不等于反证。有据反证或明确纠正不要求证明旧事日期。
4. 新增正向支持只来自本会话真实交付后的 observed_response：核对紧邻前一轮Aria实际回复及
随后本人响应，evidence与event_evidence引用该响应，只记关联、不声称因果。往事、对他人的
反应、计划、stated_preference和repeated_pattern猜测不据此新增正向人物change；原独立事件数量与状态阈值保持。

身份和更新：
- 新观察的hypothesis_id/claim_quote/scope_quote用[]；旧观察用[原id]/[完整原claim]/[完整原scope]，
不改身份或原文。review不输出id；change.review_index引用原reviews数组的0起位置，不重编号。
- change.evidence只选绑定review.evidence的原项。只有within_scope的support/counter可附change；
context_only、unknown、范围外和旧事复述不更新，不能改证据标签凑更新。
- 新change：relation=new、id=[]、lineage=[]，绑定support，claim=""交代码生成。
旧change：id/claim保持原主张，lineage=[完整原身份对象]，mechanism_relation=same、scope_relation=within；
旧support对应support，counter对应counter或retire。只有本人明确要求停止沿用旧主张才选retire。
- atomic_claim 恰有五个字段：aspect、conditions、tendency、outcome、evidence_basis。
aspect/conditions/tendency各非空且最多100字符；outcome无值[]，有值[本人有据的反应]，
最多100字符；evidence_basis与绑定review一致。stated_preference的outcome=[]，
observed_response的outcome=[本人有据的实际反应]；repeated_pattern按证据用[]或[实际反应]，不得伪造结果。
新claim由代码生成“在{conditions}的情境下，可能{tendency}”，有outcome接“；本人描述的伴随反应是{outcome}”；
总长最多180字符，不能删必要条件缩短。
- confidence为0..1有限数，表示本次证据关系强度，不是累计分，不为过门抬分；retire仍须达到0.7，否则按有效counter处理。
target_value仅big_five/disc可用[0..1数字]，其余或不明用[]；scope_tags选1..8个不同标签，general只单独用。
- framework/dimension配对：response为low_pressure/direct_options/identity_action/human_connection/meaning_memory/autonomy；
big_five为openness/conscientiousness/extraversion/agreeableness/neuroticism；disc为dominance/influence/steadiness/conscientiousness；
attachment为secure/anxious/avoidant/disorganized；cbt为core_pattern/trigger/belief/maintaining_factor/protective_factor。
big_five/disc需相对稳定行为，attachment需重要关系中的重复模式，cbt需本人可见的情境—解释/信念—情绪/行为联系。

仅Schema中的可选槽用零或一个元素的数组：review的三个身份槽、change的id/scope/lineage/target_value、
atomic_claim.outcome。无值[]，有值[值]，不写null；其余按Schema原类型。没有change也保留changes=[]。
格式示例（仅展示一个完整观察的形状，不是本次证据）：
{"protocol_version":"2.0","reviews":[{"hypothesis_id":[],"claim_quote":[],"scope_quote":[],"applicability":"unknown","relation":"unknown","evidence_basis":"context_only","evidence":[{"turn_index":0,"evidence_quote":"我想先歇一会儿。"}],"event_evidence":[{"turn_index":0,"evidence_quote":"刚整理完桌子，我想先歇一会儿。"}],"reason":"只说明当前需要休息，不据此推断稳定性格。","event_summary":"本人整理桌子后表示想休息。"}],"changes":[]}
"""


def indexed_compact_reflection_messages(messages):
    from ..memory.diagnosis import reflection_system_prompt
    if (type(messages) is not list or len(messages) != 2
            or messages[0] != {"role": "system", "content": reflection_system_prompt(True)}
            or type(messages[1]) is not dict or set(messages[1]) != {"role", "content"}
            or messages[1]["role"] != "user" or type(messages[1]["content"]) is not str):
        raise ValueError("reflection_json_requires_evidence_review_contract")
    return [{"role": "system", "content": indexed_compact_reflection_system_prompt()}, deepcopy(messages[1])]


def encode_indexed_compact_reflection(value):
    """Fixture utility only: explicit v4 encoding, never a reply repair."""
    result = encode_compact_reflection(value)
    identities = [row["id"] for row in result["reviews"]]
    if len(set(identities)) != len(identities):
        raise ValueError("ambiguous_fixture_review_id")
    for row in result["reviews"]:
        del row["id"]
    for row in result["changes"]:
        row["review_index"] = identities.index(row.pop("review_id"))
    return result


def decode_indexed_compact_reflection(raw, *, itemwise_review=False):
    """Bind original positions before itemwise rejection; never filter rows.

    Only new wire bookkeeping is projected. Illegal indices remain invalid
    changes in itemwise mode, so they cannot discard otherwise valid reviews.
    The old decoder and source validator remain authoritative for all values.
    """
    from .reflection_json import parse_reflection_json
    parsed = json.loads(parse_reflection_json(raw, itemwise_review=True))
    reviews = parsed["reviews"]
    for index, row in enumerate(reviews):
        if type(row) is dict:
            if "id" in row:
                # Preserve the complete illegal wire item as an invalid
                # item, without exposing its model ID to duplicate counting
                # or associating it with a legal sibling's generated ID.
                reviews[index] = {"invalid_wire_item": row}
                continue
            row["id"] = f"r{index + 1}"
    for proposal_index, row in enumerate(parsed["changes"]):
        if type(row) is not dict:
            continue
        if "review_id" in row:
            parsed["changes"][proposal_index] = {"invalid_wire_item": row}
            continue
        index = row.get("review_index")
        if type(index) is int and 0 <= index < len(reviews):
            del row["review_index"]
            row["review_id"] = f"r{index + 1}"
        # Missing/bool/negative/out-of-range indices cannot resolve. Keeping
        # their original field makes the unchanged local item shape reject
        # the change, without fabricating a reference or dropping the review.
    def decode_slot(item):
        if type(item) is not list or len(item) > 1 or item and item[0] is None:
            raise ValueError("invalid_compact_reflection_slot")
        return item[0] if item else None
    # Decode each item independently. A malformed slot keeps that whole item
    # invalid under the original shape contract; it cannot suppress siblings.
    for collection in ("reviews", "changes"):
        for index, row in enumerate(parsed[collection]):
            try:
                decoded = _map_slots({collection: [row]}, decode_slot)[collection][0]
            except ValueError:
                if type(row) is dict:
                    # An illegal wire scalar can happen to be a legal domain
                    # nullable value. Withhold the required code-owned link
                    # so that the item cannot masquerade as already decoded.
                    row.pop("id" if collection == "reviews" else "review_id", None)
                continue
            parsed[collection][index] = decoded
    return parse_reflection_json(json.dumps(parsed, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False), itemwise_review=itemwise_review)
