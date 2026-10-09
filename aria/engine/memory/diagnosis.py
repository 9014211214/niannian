"""会后人物假设反思。

这里不再让模型每次“重算一个人的性格”。大模型只提出本次会话对既有理解的
增量：新假设、支持、反证或撤销；确定性代码维护证据、跨会话门槛、置信度与
生命周期。变化的是 Aria 的认识，不是这个人被模型重新定义。
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any, Callable, Iterable

from ..model.persona import (
    AtomicPersonClaim,
    EvidenceReviewQuote,
    PersonEvidenceReview,
    PersonHypothesis,
    PersonHypothesisEvidence,
    PersonHypothesisTransition,
    PersonaProfile,
)
from .belief_change import (
    CONFIDENCE_UPDATE_POLICY_VERSION,
    belief_state,
    update_confidence,
)
from .reflection_contract import reflection_response_schema as reflection_response_schema

_FRAMEWORKS = {"big_five", "disc", "attachment", "cbt", "response"}
_RELATIONS = {"new", "support", "counter", "retire", "no_change"}
_ATTACHMENT = {"secure", "anxious", "avoidant", "disorganized"}
_BIG_FIVE = {
    "openness", "conscientiousness", "extraversion",
    "agreeableness", "neuroticism",
}
_DISC = {"dominance", "influence", "steadiness", "conscientiousness"}
_CBT_DIMENSIONS = {
    "core_pattern",
    "trigger",
    "belief",
    "maintaining_factor",
    "protective_factor",
}
_RESPONSE_DIMENSIONS = {
    "low_pressure",
    "direct_options",
    "identity_action",
    "human_connection",
    "meaning_memory",
    "autonomy",
}
_FRAMEWORK_DIMENSIONS = {
    "big_five": _BIG_FIVE,
    "disc": _DISC,
    "attachment": _ATTACHMENT,
    "cbt": _CBT_DIMENSIONS,
    "response": _RESPONSE_DIMENSIONS,
}


def is_conversation_quote_source(source: str) -> bool:
    """Recognise updater-owned provenance formats, not evidence truth.

    Consumers must still verify the cited session, turns and exact user quote.
    A lineage annotation preserves the original source domain; arbitrary
    suffixes, unknown labels and near-prefix matches are not accepted.
    """
    if source == "conversation_quote":
        return True
    prefix = "conversation_quote:lineage:"
    if not isinstance(source, str) or not source.startswith(prefix):
        return False
    parts = source[len(prefix):].split("/")
    return len(parts) == 2 and parts[1] in _FRAMEWORK_DIMENSIONS.get(parts[0], set())


_REFLECTION_OBSERVATION_MODALITIES = {
    "acoustic",
    "behavioral",
    "context",
    "environment",
    "prosodic",
    "sensor",
    "visual",
    "wearable",
}
_SCOPE_TAGS = {
    "general",
    "concealed_distress",
    "reduced_life_engagement",
    "loss_of_role",
    "social_disconnection",
    "physical_discomfort",
    "autonomy_threat",
    "grief_or_transition",
    "task_request",
    "relationship_rupture",
}
_SYSTEM_PROMPT = """\
你负责会后人物理解反思。你不是在给人下诊断，也不要每次重算一整套人格分数。

先判断本次是否真的产生了长期人物证据。客观事实、第三人的偏好、一次性情绪、
玩笑、年龄刻板印象和 Aria 自己的解释都不是长期人物证据；这些情况返回
changes=[]。若有证据，只返回一个对未来交互最有用、证据最强的 change。不要
为了覆盖框架输出第二个相邻解释；输出数组即使收到多个候选也只能保留第一名。

五类框架必须按“证据实际说明了什么”严格分开：
- response：某种沟通、支持或互动方式之后，出现了用户可观察的接受、拒绝、
  放松、退出、参与等结果。必须同时有“方式”和“结果”，不能只因用户有某种
  行为或偏好就归为 response。六个 dimension 的主因只能选一个：
  * low_pressure：核心变量是少追问、少催促、停顿或留白；
  * direct_options：核心变量是少量、清楚、可执行的选项；
  * identity_action：核心变量是经验、能力或旧角色被真实用于行动；
  * human_connection：核心变量是与谁、以何种人际安排建立连接；
  * meaning_memory：核心变量是个人经历、熟悉媒介或回忆成为安全入口；
  * autonomy：核心变量是同意、选择权和控制权本身。
- Big Five：用户自己在不同任务或情境中重复出现的相对稳定行为倾向。提前
  计划、主动组织、持续探索等属于行为倾向，而不是 response；单一事件只能是
  candidate，跨独立会话重复后才支持。dimension 使用五因素英文名。
- DISC：用户在协作、沟通和做事中的风格启发；不得与 Big Five 重复描述同一
  行为，也不是临床人格诊断。
- attachment：只用于特定重要关系中反复出现的接近、回避、安全感或被抛弃
  模式；普通社交人数偏好不等于依恋类型。
- CBT：必须能从用户原话建立明确的“情境/触发 → 信念或解释 → 情绪或行为”
  链。仅有情绪、行为或某句格言不足以形成 CBT 假设；它是动态个案假设，不是
  人格类别。

同一组证据只归入最贴近的一个框架和一个 dimension。先检查既有假设的主张、
作用机制和适用范围，再决定关系；不要先从本轮某个词重新选择人格标签。
人物身份是已限定的主张与机制，不是 framework/dimension 标签。生成 change 前，
先逐一对照既有主张（包括 candidate），再分类：本轮是否还是同一种方式导致
同一种可观察结果？变化的只是措辞/媒介，还是实际起效机制和适用范围变了？
不能因本轮同时提到选择、时间或材料，就跳过原假设重新建档。即使你现在会把
原机制命名为另一个 dimension，也应保留原主张的 id 与标签；重新分类本身不
是新证据，不得自动合并已存在的两个不同主张。无法确定同一机制时不要强行复用。
先找最具体、实际导致结果的变量，不能把低压力、选择权、关系连接等相邻解释
同时建档。选择某种互动的具体材料、时间或形式，不自动证明“选择权本身”是
起效机制；只要仍是既有方式产生相应结果，就沿用原 framework、dimension、id。
若已有同一 framework、dimension 和适用情境的假设，必须引用其 id 做
support/counter/retire，不要另建同义假设。若既有理解仍成立但没有新证据，
返回 changes=[]；
区分以下三种情况，不能把限制条件一律当成反证：
1. 既有主张在原适用范围内再次出现，包括说明什么条件使同一方式有效或无效，
   且与原主张一致，relation 是 support，引用原 id。负面结果若发生在该方式
   缺失或被反向实施时，可能支持原倾向，而不是推翻它；须读完整条件和结果。
2. 本轮在既有适用范围内出现与主张相反的结果，或证明原主张/范围确实过宽，
   relation 必须是 counter，并引用既有 id。只记录反例情境，不把相反主张当
   成旧假设的新名称，也不能用相邻框架掩盖反证。
3. 原假设已明确限定范围，本轮只谈范围以外的不同情况，既无范围内支持也无
   范围内反证，则 changes=[]。不能凭“并非所有情况如此”罚掉已限定的主张，
   也不能为了记录边界而虚构一次支持。若同时存在支持与真正反证，优先真实
   反证，不将两种方向混成一条支持证据。
恢复阶段也按原主张及已知反例检查；原模式在其适用范围内再次真实出现时，
引用原 id 返回 support，不因新的措辞、媒介或局部选择而拆成相邻假设。当前人物
假设中的 support_sessions、counter_sessions、status 和 recent_relations 是历史
账本摘要；lineage_anchors 是同一主张最早留存的支持、最近支持和最近反证，只用于
比较本轮证据与旧主张的关系，不能把历史本身当作本次新证据。
retired 假设仍可被重新检验：本次确有支持同一主张和适用范围的用户原话时，
引用原 id 返回 support，不要另建同名假设。它会重新从 candidate 积累独立会话
证据，退休前的支持不会直接恢复其在线影响。反例情境只记录为反证的范围，
不能把反例情境合并为原假设的新适用范围。
若有证据，relation 只能是 new/support/counter/retire。每条必须引用本次对话的
turn_index，并逐字复制该轮用户原话中的 evidence_quote；解释性改写、Aria 的
回复和在线模型标签都不能作为证据。把适用情境写进 scope。不要从年龄、性别
或一次情绪泛化人格。
在线状态、情绪候选和干预选择不会提供给会后反思，也不能作为人格证据；只能
依据对应 turn_index 的用户原话、后续真实反应和真实多模态观测。若原话不足
或有反证，返回 no change 或 counter，不能生成长期人物结论。

已有主张的 support/counter/retire 还须填写 lineage：逐字复制原 id、claim、scope
（原 scope 为空则复制空字符串），mechanism_relation=same、scope_relation=within。
这表示已比较同一机制和原适用范围，并非因为标签相近而猜测。反证可以在同一
机制下产生相反结果，仍然填 same/within 并用 relation=counter。原主张范围外
或不同机制不能填此 lineage；有独立新证据时用 new 且 lineage=null，无新证据
则 changes=[]。保留原主张，当前情境另写在 change.scope，不能用它扩大原范围。

只输出 JSON：
{"changes":[{
  "id":"更新已有假设时必须填其 id；新假设留空",
  "framework":"big_five|disc|attachment|cbt|response",
  "dimension":"必填；Big Five/ DISC 填对应英文维度；attachment 填类型；CBT 填 core_pattern|trigger|belief|maintaining_factor|protective_factor",
  "claim":"可推翻的中文假设",
  "scope":"它可能只在哪类情境成立",
  "scope_tags":["从 general、concealed_distress、reduced_life_engagement、loss_of_role、social_disconnection、physical_discomfort、autonomy_threat、grief_or_transition、task_request、relationship_rupture 中选择"],
  "relation":"new|support|counter|retire",
  "lineage":{"id":"原假设 id","claim_quote":"逐字复制原 claim","scope_quote":"逐字复制原 scope 或空字符串","mechanism_relation":"same","scope_relation":"within"}或null,
  "target_value":0到1或null,
  "confidence":0到1,
  "evidence":[{"turn_index":0,"evidence_quote":"逐字复制该轮用户原话"}]
}]}
"""


EVIDENCE_REVIEW_PROTOCOL_VERSION = "2.0"
EVIDENCE_REVIEW_PROMPT_CONTRACT_VERSION = "2.4"
# Public diagnostic values are codes, never model output or exception messages.
# Keep this explicit: new validators default to the generic category until their
# non-sensitive code has deliberately been reviewed and added here.
_REFLECTION_ERROR_DETAILS = frozenset({
    "model_call_failed", "invalid_json_response", "invalid_reflection_shape", "invalid_evidence_review_protocol",
    "reflection_object_required", "nonfinite_json_number", "changes_array_required",
    "review_protocol_version_required", "review_or_change_bound_exceeded",
    "invalid_review_fields", "duplicate_or_invalid_review_id",
    "new_review_has_unbound_prior_text", "review_prior_identity_mismatch",
    "invalid_scope_relation", "invalid_review_relation", "invalid_evidence_basis",
    "invalid_event_kind", "unknown_prior_event_reference",
    "same_event_requires_known_reference", "new_event_cannot_reference_old_event",
    "invalid_bounded_review_explanation",
    "event_reference_disagrees_with_known_quote_signature",
    "counter_requires_existing_hypothesis", "change_requires_exact_review_binding",
    "one_review_cannot_duplicate_changes", "change_evidence_basis_mismatch",
    "change_quote_not_in_bound_review", "change_relation_disagrees_with_review",
    "existing_support_requires_support_change", "invalid_change_confidence",
    "invalid_explicit_scope_tags", "invalid_or_inapplicable_person_scale",
    "new_claim_requires_unbound_identity", "existing_claim_requires_explicit_review_identity",
    "change_existing_claim_identity_mismatch", "invalid_reviewed_change_shape",
    "bounded_current_quotes_required", "invalid_review_quote_shape",
    "review_quote_not_grounded_in_current_user", "one_atomic_claim_required",
    "invalid_atomic_claim_text", "context_or_authorization_is_not_person_claim",
    "invalid_atomic_outcome", "stated_preference_must_not_invent_causal_outcome",
    "observed_response_requires_reported_outcome", "atomic_claim_exceeds_retained_claim_bound",
})
_REVIEW_LEDGER_LIMIT = 40
_REVIEW_PROMPT_LIMIT = 24
_EVIDENCE_SYSTEM_PROMPT = """你负责会后人物理解，不做在线聊天、不诊断、不授予任何工具权限。
本次严格使用 evidence-review protocol 2.0：先依据用户原话审查证据，再至多更新一条
对未来互动有用的、可推翻的原子人物主张。只能输出规定 JSON，不输出思维过程。
本提示的输出合同为 prompt contract 2.4；数据 protocol_version 仍必须为字符串 "2.0"。

重要区别：
1. 范围内支持、范围内真正反例、范围以外和未知是不同结果。已有主张先逐字对照其
claim 和 scope，再判 applicability。新主张没有旧身份：先写有本人证据且范围有限的
atomic_claim.conditions，再相对这条新主张的条件判断 applicability；没有旧 hypothesis
本身不是 unknown。本人明确在这些条件下表达偏好，可为 within_scope support 并形成
低置信候选，不需要先有旧主张。若证据的主体、条件或相关性确实不明，仍用 unknown、
changes=[]，不能为形成标签强改 within_scope；范围尚未外推也不等于连局部证据都未知。
无 change 的新观察若没有明确待检验条件，可以保存 unknown review，无需补造主张。
噪声、忙其他事情解释短答，若旧主张只说安静且
空闲的场景，这是 outside_scope，不是 counter。没有试过、不知道某方式是否舒服，
对该方式的偏好仍是 unknown，不是已经证明相反偏好。不愿被他人喜好代表也不能推出
“喜欢自己先试再选”或一般自主性格；如果本人明确提出“别代我选”，最多保留该次同类
选择沟通中不接受替选的局部表达，不补成其他学习/活动的稳定偏好。
不得用“撤回谨慎一点”替代正确的证据关系。
2. 本人明确说“更喜欢怎么交流”本身就是 stated_preference 的合法证据，不要求你
编造方式导致放松/接受等结果。observed_response 必须同时有方式与本人反应；只能写
观察到的关联，不能把关联说成因果。repeated_pattern 必须来自多个独立事件；多句话、
多次转录、同一往事反复讲、照片和文字呈现同一事，都不是多个独立事件。
3. 首次提起自己的往事可以保留一次 retrospective 候选上下文；不能当稳定人格证明。
再次读同一日记、引用旧聊天、谈同一事件的另一媒介，要链接旧 event_id 并用 same_event，
不能刷新支持会话或置信。event_evidence 应引用识别事件的上下文，而非只挑一句偏好。
不知道是不是同一件事时用 unknown。旧观察中“不采纳/未知”的原因也要阅读，它们不会
因为上次没有形成假设而变成不存在。旧观察的reason和summary是系统评估，不是本人事实。
本人现在明确纠正你对旧事的理解，是新的解释/偏好证据而不是旧行为再次发生；用new_event，
event_evidence引用本次明确纠正的原话，不能只是重贴旧事件原句。单纯复述不是纠正。
附属 context_only review 也遵守相同事件合同：同一轮里提到旧事，不代表当前的纠正或
授权声明就是 same_event。当前新声明用 new_event、prior_event_id=null；实际复述已知
旧事件才用 same_event 并逐字复制已提供的 event_id；首次叙述往事用 retrospective，
未能确定是否旧事件用 unknown、prior_event_id=null。没有已知 event_id 时不得填 same_event。
不要把 review.id、hypothesis_id 或当前新编的标识当作 prior_event_id，也不要省略无法
引用的附属 review 来掩盖证据错误；只提交有当前逐字出处且事件关系能如实表达的观察。
同一批 reviews 中首次描述的一个新事件可以有多条观察：它们均用 new_event、
prior_event_id=null，并逐对象复用同一组 event_evidence 引用（turn_index 和
evidence_quote 完全一致），系统会给它们同一个 event_id，不把多条观察算成多次
独立事件。same_event 专指请求中已提供 event_id 的历史事件，不能表示“与本批
r1/r2是同一件事”，也不能把当前 review.id 当旧 event_id。不确定是否同一事件
仍用 unknown；不能为合组捏造共同背景，也不能把已知旧事件改称新事件。
首次行为证据可提出 repeated_pattern 方向的暂定候选，但只能说“可能”，不能宣称重复
模式已被观察证实；系统仍要求独立事件数量与原有阈值才允许成为工作/稳定假设。
4. 一个 atomic_claim 只包含一个 aspect、其 conditions、一个 tendency。outcome 字段必须
存在，按证据类型填字符串或 JSON null，不能省略；不是所有证据类型都需要实际结果。
stated_preference 的 outcome 必须为 null。不要把“先问本人”“允许外发”“会更愿意
表达”三个不同命题混成一个人物结论。授权、撤销授权、医疗安全规则不是推断的人格；
任何偏好都不能等同于本次已同意联系家属/外发。单纯授权操作用 context_only，无人格change。
5. 真正相关的本人纠正优先于旧推断。旧主张包含一个被本人否定的部分，即便另一部分
仍成立，也要为受影响的同一主张记录 within_scope counter，不能因另一部分继续成立
就把整体标为 support。不要修改旧claim为相反观点来伪装同ID连续性。
6. 相同主张以 claim/scope 为身份，不以心理学标签为身份。已有主张必须给 hypothesis_id、
claim_quote、scope_quote，并填写匹配它的 lineage。范围外/未知不得填写“within”的更新。
有反例且有支持时优先真正反例；最多一个 change，但可以记录最多六个有来源的 reviews。
reviews 可以没有 change：保存已引用的未知、无关、复述和不更新理由，不需要为了入账造标签。
只依据当前 turn_index 的用户原话，不能引用 Aria 推测、他人/影视角色的偏好为本人偏好。
不能从年龄、性别或一次情绪推人格，不把风险规则/授权策略包装成个性。
7. 最小推断范围：conditions 必须保留原话限定的事情、关系、紧急性和阶段；不是要求
逐字复述，但不能把一次非紧急失落时希望先倾诉，扩成任何事都不找家人；不能把这次
手艺展示中的愿意参与，扩成所有集体活动都积极。当前证据不足以支撑更大范围时，先
保留有据的局部候选，或 unknown review；reason 简短说明保留了哪些限制以及哪些
尚未知。没有候选同样是合法结果，不能从未知话术反推出一个“谨慎型”标签。
8. confidence 是你对本次逐字证据支持/反驳所声明主张的强度评估，不是人物假设更新后
应达到的分数，不是这个人“有几成这种性格”。已有假设的 confidence 是系统累计的
工程信念分，不能直接复制为本次 confidence。系统同时考虑新事件、证据和生命周期；
这些数字不是经过心理测量校准的概率。普通新反例用 counter。只有证据明确要求停止
沿用该旧主张时才提 retire；retire 且 confidence >= 0.7 执行直接撤回，低于 0.7 按
有效 counter 降低信念，不会忽略已采纳的反证。不要为了越过阈值抬高 confidence；
是否最终撤回还取决于系统状态。证据入账、信念实际变化、实际撤回是三个不同结果。

framework 输出只选小写枚举 response/big_five/disc/attachment/cbt，不能写 Big Five、
BigFive、DISC 或 CBT；dimension 也使用下面的小写枚举，不输出中文框架或维度名称。
框架：response用于用户明确的沟通偏好或实际互动反应，维度 low_pressure/direct_options/
identity_action/human_connection/meaning_memory/autonomy，按一个主要方面选。
big_five 的维度 openness/conscientiousness/extraversion/agreeableness/neuroticism；
disc维度 dominance/influence/steadiness/conscientiousness，均需相对稳定的行为依据。
attachment维度 secure/anxious/avoidant/disorganized，只用于重要关系中的重复模式。
cbt维度 core_pattern/trigger/belief/maintaining_factor/protective_factor，需要用户可见的
情境—解释/信念—情绪/行为联系，不从行为表面编造深层心理。

逐项输出限制（字符数按字符串长度计，不是 token 数；违反时本次结果整体不入账）：
- 顶层只能有 protocol_version/reviews/changes；各对象必须完整保留以下 JSON 形状中的
  全部字段，不能增删字段。不要加 Markdown 围栏、前后说明、注释或推理过程。
- reviews 为数组，可为空，最多 6 项；changes 只提出 0 或 1 项，不使用 no_change 对象。
  每个 change 必须绑定一个实际 review.id，同一个 review 不得重复用于多个 changes。
- review.id 必须是 1..40 个 ASCII 字母、数字、下划线或短横线组成的字符串，本次不重复。
  新主张 review.hypothesis_id/claim_quote/scope_quote 均为 JSON null，不能写字符串 "null"。
  已有主张的这三项必须逐字匹配提供的旧 id/claim/scope；旧 scope 为 null 时 scope_quote
  用空字符串。不能编造已有 id。所有枚举必须从形状列出的值中选一个，不输出竖线组合。
- prior_event_id 为 JSON null 或已提供的 event_id（不是 review.id）；same_event 必须有
  旧 event_id，new_event 必须为 null。不确定事件身份用 unknown，不编造旧事件引用。
  完全相同的 event_evidence 引用集合也会被识别为同一旧事件，不会刷新证据计数。
- 每个 review.evidence、review.event_evidence 和 change.evidence 都是 1..3 项数组。
  每项只能有 turn_index/evidence_quote；turn_index 是当前非排除用户轮次的整数索引，
  不能是布尔值。evidence_quote 是该轮用户原话中连续逐字子串，长度 2..240 个字符；
  同一数组内 (turn_index,evidence_quote) 不重复，不可用空数组、改写或助手原话代替。
- change.evidence 可选用 review.evidence 中的部分条目，但每个引用对象必须完全复制：
  turn_index 与 evidence_quote 都逐字相同。不能缩短成另一子串、改写、拼接或另加引用。
- reason 和 event_summary 必须为非空、非纯空白字符串，每项最多 240 个字符。
- atomic_claim 的 aspect/conditions/tendency 必须为非空、非纯空白字符串，各最多 100
  个字符。outcome 若为字符串也必须非空、非纯空白且最多 100 个字符；stated_preference
  必须为 null，observed_response 必须为用户有据的反应字符串，repeated_pattern 可为 null。
  atomic_claim.evidence_basis 必须与绑定 review 一致，不能为 context_only。
- 新主张由系统拼成“在{conditions}的情境下，可能{tendency}”；有 outcome 时再接
  “；本人描述的伴随反应是{outcome}”。包括这些固定文字在内的总长度不得超过 180 个字符，
  各字段分别未超 100 也不代表拼接后合格。不要为缩短而删除必要的范围限制。
- scope_tags 为 1..8 个不重复的下列白名单字符串；general 只能单独使用，不能与任何
  情境标签并列。confidence 必须是 0..1 的有限 JSON 数字，不接受布尔值、NaN 或 Infinity。
  target_value 仅 big_five/disc 可填 0..1 的有限数字，否则必须为 JSON null；不确定时用 null。
- 新主张 change.id 用空字符串或 JSON null、lineage=null、relation=new，绑定 support
  review。已有 id 的 support review 只允许 change.relation=support，不能使用 new；
  已有主张 counter review 对应 counter 或 retire。已有 claim 必须原样保留，lineage 的
  id/claim_quote/scope_quote 必须完整匹配旧主张且 mechanism_relation=same、scope_relation=within。
  不应更新的范围外、未知、复述或 context_only review 不要附带 change。
  新主张的 within_scope 是证据与 atomic_claim.conditions 相符，不是它已有历史；
  unknown 与 relation=new 的 change 同时输出不会入账，审查完成也不代表人物已经学会。
- 输出前核对绑定：新主张的 change 只能对应 within_scope/support 的 review；已有
  主张的 change 对应 within_scope/support 或 counter。unknown、outside_scope、
  context_only、旧事件复述等观察可保留 review，但不附 change。不把 unknown 改成
  support 来凑出 change，也不将 stated_preference 的实际反应强塞入 outcome；若
  证据类型与结果字段无法如实一致，保留观察、changes=[]。

严格 JSON 形状（即使不更新也必须有protocol_version和reviews）：
{"protocol_version":"2.0","reviews":[{
"id":"r1","hypothesis_id":"已有id或null","claim_quote":"旧claim或null",
"scope_quote":"旧scope逐字复制，空范围用空字符串；新主张null",
"applicability":"within_scope|outside_scope|unknown","relation":"support|counter|neutral|unknown",
"evidence_basis":"stated_preference|observed_response|repeated_pattern|context_only",
"event_kind":"new_event|same_event|retrospective|unknown","prior_event_id":"已知事件id或null",
"evidence":[{"turn_index":0,"evidence_quote":"当前用户逐字原话"}],
"event_evidence":[{"turn_index":0,"evidence_quote":"当前原话中的事件背景或当前纠正"}],
"reason":"简短说明为何采纳或不采纳，只是评估不是新增事实",
"event_summary":"简短事件识别说明，不写未经证实的细节"
}],"changes":[{
"review_id":"r1","id":"已有id；新主张空字符串","framework":"response",
"dimension":"low_pressure","claim":"旧claim原样；新主张会按atomic_claim生成",
"scope":"适用情境","scope_tags":["general"],"relation":"new|support|counter|retire",
"lineage":{"id":"旧id","claim_quote":"旧claim","scope_quote":"旧scope",
"mechanism_relation":"same","scope_relation":"within"},
"atomic_claim":{"aspect":"一个沟通或行为方面","conditions":"特定场景",
"tendency":"可推翻的单一倾向，不含外发授权或另一个命题","outcome":null,
"evidence_basis":"stated_preference"},"target_value":null,"confidence":0.6,
"evidence":[{"turn_index":0,"evidence_quote":"必须在所绑定review.evidence内"}]
}]}
新主张 lineage=null；无适用人物变化用 changes=[]。所有引用都必须来自当前非排除的用户轮次。
scope_tags只选 general/concealed_distress/reduced_life_engagement/loss_of_role/
social_disconnection/physical_discomfort/autonomy_threat/grief_or_transition/task_request/
relationship_rupture。new_event无旧event_id，same_event必须链接已提供的旧event_id。
"""


def reflection_system_prompt(evidence_review_protocol: bool = False) -> str:
    return _EVIDENCE_SYSTEM_PROMPT if evidence_review_protocol else _SYSTEM_PROMPT


def _safe_reflection_error_detail(error: Exception, fallback: str) -> str:
    """Return only a reviewed constant; never stringify untrusted exceptions."""
    if (type(error) is ValueError and len(error.args) == 1
            and type(error.args[0]) is str and error.args[0] in _REFLECTION_ERROR_DETAILS):
        return error.args[0]
    return fallback if fallback in _REFLECTION_ERROR_DETAILS else "invalid_evidence_review_protocol"


def _review_context(persona: PersonaProfile) -> list[dict]:
    """Bounded offline-only observations; non-adoption is not forgotten immediately."""
    ledger = persona.evidence_review_ledger[-_REVIEW_LEDGER_LIMIT:]
    # Preserve a few early non-adoptions alongside recent observations, so an
    # old unknown is not replaced solely by a later one-sided claim summary.
    anchors = []
    seen_targets = set()
    for row in ledger:
        target = row.hypothesis_id or row.event_id
        if row.disposition != "applied" and target not in seen_targets:
            anchors.append(row)
            seen_targets.add(target)
        if len(anchors) >= 6:
            break
    selected = {row.id: row for row in [*anchors, *ledger[-18:]]}
    return [row.model_dump() for row in selected.values()][-_REVIEW_PROMPT_LIMIT:]


def build_reflection_messages(
    persona: PersonaProfile, conversation: str, observations: list[dict],
    evidence_review_protocol: bool = False,
) -> list[dict[str, str]]:
    """The exact input builder shared with offline audit; never called online."""
    from .hypothesis_context import reflection_hypotheses

    existing = [{
        "id": item.id, "framework": item.framework, "dimension": item.dimension,
        "claim": item.claim, "scope": item.scope, "scope_tags": item.scope_tags,
        "confidence": round(item.confidence, 3), "status": item.status,
        "support_sessions": len(item.supporting_sessions()),
        "counter_sessions": len(item.counter_sessions()), "activation_epoch": item.activation_epoch,
        "activation_support_sessions": len({e.session_id for e in item.activation_evidence() if e.relation == "support"}),
        "lineage_anchors": _lineage_anchors(item, include_review=evidence_review_protocol),
        "recent_relations": [{
            "relation": e.relation, "at": e.at, "summary": e.summary,
            "scope": e.scope, "scope_tags": e.scope_tags,
        } for e in item.evidence[-4:]],
        **({"atomic_claim": item.atomic_claim.model_dump() if item.atomic_claim else None,
            "support_event_ids": sorted(item.supporting_events())} if evidence_review_protocol else {}),
    } for item in reflection_hypotheses(persona)]
    content = (
        "当前人物假设：\n" + json.dumps(existing, ensure_ascii=False)
        + "\n\n本次会话：\n" + conversation
        + "\n\n本次可选多模态观测（缺失模态不是反证）：\n" + json.dumps(observations, ensure_ascii=False)
        + "\n\n在线状态、情绪候选和干预选择不会提供给会后反思；"
        "只能依据上面的用户原话与真实多模态观测。"
    )
    if evidence_review_protocol:
        content += "\n\n过去的有界证据审查（含未知/不采纳；不是本次新证据或授权）：\n" + json.dumps(
            _review_context(persona), ensure_ascii=False,
        )
    from .stated_preferences import confirmed_preference_payloads
    preferences = confirmed_preference_payloads(persona)
    if preferences:
        content += (
            "\n\n本人确认的相处说明（只读历史记录，不是本次新证据）：\n"
            + json.dumps(preferences, ensure_ascii=False)
            + "\n不得从这些记录重复累计证据、支持轮次、事件或置信度；"
            "不能只凭它们新建、支持或否定人物假设。必须核对上方本次会话的本人原话。"
            "不得作出与本人确认的说明相反的泛化推断；如本次出现直接反证，应记录适用范围与冲突，"
            "不得替本人撤回或改写这些说明，也不因此取得分享、通知等行动权限。"
        )
    return [{"role": "system", "content": reflection_system_prompt(evidence_review_protocol)},
            {"role": "user", "content": content}]


def _strip_fences(raw: str) -> str:
    value = raw.strip()
    if value.startswith("```"):
        value = value.split("\n", 1)[1] if "\n" in value else value[3:]
        if "```" in value:
            value = value[: value.rfind("```")]
    return value.strip()


def _parse_json_object(raw: str) -> dict:
    """Accept a JSON object with harmless fences/preamble, never repair data."""
    value = _strip_fences(str(raw or ""))
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        start = value.find("{")
        if start < 0:
            raise
        parsed, _ = json.JSONDecoder().raw_decode(value[start:])
    if not isinstance(parsed, dict):
        raise ValueError("reflection response must be a JSON object")
    return parsed


def _unwrap_review_response(raw: str) -> tuple[str, str]:
    """Remove one complete JSON fence, never search for or repair an object.

    This is an explicitly audited transport accommodation for protocol 2.0
    since prompt contract 2.2. Legacy parsing and independent judge parsing are
    intentionally untouched. The caller keeps the original response unchanged.
    """
    if not isinstance(raw, str):
        raise ValueError("invalid_json_response")
    value = raw.strip()
    match = re.fullmatch(r"```json[ \t]*\r?\n([\s\S]*?)\r?\n```", value)
    if match is not None:
        body = match.group(1)
        # A second Markdown delimiter is not a single transport wrapper. JSON
        # containing escaped backticks inside a string is not a delimiter.
        if re.search(r"(?m)^[ \t]*```", body):
            raise ValueError("invalid_json_response")
        return body, "single_json_fence"
    return value, "bare_json"


def _clamp(value: Any, default: float = 0.5) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _compact(value: Any, limit: int = 180) -> str:
    return " ".join(str(value or "").strip().split())[:limit]


def _normalise_scope_tags(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    tags = list(dict.fromkeys(
        str(item).strip().lower()
        for item in value[:8]
        if str(item).strip().lower() in _SCOPE_TAGS
    ))
    # general 不能和情境标签同时出现，否则局部偏好会伪装成全局人格。
    if "general" in tags and len(tags) > 1:
        tags.remove("general")
    return tags


def _similarity_text(value: Any) -> str:
    """Normalise punctuation without pretending to do semantic inference."""
    return re.sub(r"[^0-9a-z\u3400-\u9fff]+", "", str(value or "").lower())


def _character_ngrams(value: str, size: int = 2) -> set[str]:
    if len(value) < size:
        return {value} if value else set()
    return {value[index:index + size] for index in range(len(value) - size + 1)}


def _text_similarity(left: Any, right: Any) -> float:
    """Conservative surface similarity for same-dimension paraphrase merging.

    This is deliberately not an embedding or a second model call.  It may miss a
    distant paraphrase, but cannot silently turn semantic guesses into identity.
    """
    a = _similarity_text(left)
    b = _similarity_text(right)
    if not a or not b:
        return 1.0 if a == b else 0.0
    if a == b:
        return 1.0
    grams_a = _character_ngrams(a)
    grams_b = _character_ngrams(b)
    dice = (
        2 * len(grams_a & grams_b) / (len(grams_a) + len(grams_b))
        if grams_a and grams_b else 0.0
    )
    sequence = SequenceMatcher(None, a, b, autojunk=False).ratio()
    return max(dice, sequence)


def _scope_tag_similarity(left: Iterable[str], right: Iterable[str]) -> float:
    a = set(left)
    b = set(right)
    if not a or not b:
        return 0.0
    if "general" in a | b:
        return 1.0 if a == b == {"general"} else 0.0
    return len(a & b) / len(a | b)


def _hypothesis_id(
    framework: str, dimension: str | None, claim: str, scope: str | None
) -> str:
    raw = "|".join((
        framework,
        dimension or "",
        _compact(claim).lower(),
        _compact(scope).lower(),
    ))
    return "ph_" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def _lineage_anchors(hypothesis: PersonHypothesis, *, include_review: bool = False) -> list[dict]:
    """Bounded offline references preserve meaning when recent rows are one-sided.

    These are historical, grounded ledger entries, not new observations or a
    model-generated summary. Earliest retained support remains available after
    several counterexamples; an old epoch never earns fresh activation. The
    persisted ledger's existing retention bound is not changed here.
    """
    supports = [
        item for item in hypothesis.evidence if item.relation == "support"
    ]
    counters = [
        item for item in hypothesis.evidence if item.relation == "counter"
    ]
    candidates = [
        ("earliest_retained_support", supports[0] if supports else None),
        ("latest_support", supports[-1] if supports else None),
        ("latest_counter", counters[-1] if counters else None),
    ]
    anchors: list[dict] = []
    seen: set[tuple] = set()
    for purpose, evidence in candidates:
        if evidence is None:
            continue
        key = (
            evidence.session_id, evidence.relation,
            tuple(evidence.turn_indices), evidence.summary,
        )
        if key in seen:
            continue
        seen.add(key)
        anchors.append({
            "purpose": purpose,
            "session_id": evidence.session_id,
            "turn_indices": evidence.turn_indices,
            "relation": evidence.relation,
            "summary": evidence.summary,
            "scope": evidence.scope,
            "scope_tags": evidence.scope_tags,
            "activation_epoch": evidence.activation_epoch,
            **({"event_id": evidence.event_id, "review_id": evidence.review_id,
                "event_kind": evidence.event_kind, "evidence_basis": evidence.evidence_basis}
               if include_review else {}),
        })
    return anchors


class PersonaHypothesisUpdater:
    """把会话证据更新为可支持、可反驳、可撤销的长期人物假设。"""

    def __init__(
        self,
        llm,
        min_turns: int = 3,
        now_provider: Callable[[], datetime] | None = None,
        evidence_review_protocol: bool = False,
        body_json_protocol: bool = False,
    ):
        self.llm = llm
        self.min_turns = min_turns
        self._now = now_provider or datetime.now
        self.last_status = "not_run"
        self.last_error_type: str | None = None
        self.last_error_detail: str | None = None
        self.evidence_review_protocol = evidence_review_protocol
        if type(body_json_protocol) is not bool or body_json_protocol and not evidence_review_protocol:
            raise ValueError("body_json_requires_evidence_review_protocol")
        self.body_json_protocol = body_json_protocol
        self.last_format_attempts: list[dict] = []
        self.last_format_correction_count = 0
        self.last_review_summary: dict | None = None
        self.last_response_transport: dict | None = None
        self.last_proposal_results: list[dict] = []
        self.last_application_summary: dict | None = None
        # None means not evaluated/failed, [] means completed without a fresh
        # applied change. These are attempt-local effects, not committed truth.
        self.last_belief_change_results: list[dict] | None = None

    def update(self, persona: PersonaProfile, ctx) -> PersonaProfile:
        self.last_review_summary = None
        self.last_error_detail = None
        self.last_response_transport = None
        self.last_proposal_results = []
        self.last_application_summary = None
        self.last_belief_change_results = None
        self.last_format_attempts = []
        self.last_format_correction_count = 0
        state = persona.person_model_state
        if ctx.session_id in state.reflected_session_ids:
            self.last_status = "skipped"
            self.last_error_type = None
            return persona
        if len(ctx.turns) < self.min_turns:
            self.last_status = "skipped"
            self.last_error_type = None
            return persona

        excluded = _sensitive_memory_excluded_turns(ctx)
        conversation = "\n".join(
            f"[turn {index}] 用户:{turn.user}\nAria:{turn.agent}"
            for index, turn in enumerate(ctx.turns)
            if index not in excluded
        )
        observations = [
            {
                "turn_index": item.turn_index,
                "summary": item.summary(),
            }
            for item in ctx.reasoning.evidence
            if (
                item.modality in _REFLECTION_OBSERVATION_MODALITIES
                and item.turn_index not in excluded
                and not item.fallback
                and item.weight >= 0.25
            )
        ][-20:]
        messages = build_reflection_messages(
            persona, conversation, observations, self.evidence_review_protocol,
        )
        self.last_error_type = None
        try:
            raw = self._body_json_response(messages) if self.body_json_protocol else self.llm.complete(messages)
        except Exception:
            self.last_status = "failed"
            self.last_error_type = "model_call_failed"
            self.last_error_detail = "model_call_failed"
            if self.evidence_review_protocol:
                self._record_proposal_outcomes(None, {}, review_status="not_evaluated")
            return persona
        if raw is None and self.body_json_protocol:
            self.last_status = "failed"
            self.last_error_type = "invalid_json_response"
            # Keep the terminal type used by existing queue/control gates while
            # distinguishing parsed-but-invalid shape from JSON syntax. Only
            # the safe validator category is inspected, never raw model text.
            self.last_error_detail = (
                "invalid_reflection_shape"
                if self.last_format_attempts
                and self.last_format_attempts[-1].get("shape", {}).get("kind") == "schema"
                else "invalid_json_response"
            )
            self._record_proposal_outcomes(None, {}, review_status="not_evaluated")
            return persona
        try:
            if self.evidence_review_protocol:
                self.last_response_transport = {
                    "prompt_contract_version": EVIDENCE_REVIEW_PROMPT_CONTRACT_VERSION,
                    "format": "not_evaluated", "wrapper_removed": False,
                    "json_status": "not_evaluated", "content_repaired": False,
                }
                json_text, response_format = _unwrap_review_response(raw)
                self.last_response_transport.update(
                    format=response_format,
                    wrapper_removed=response_format == "single_json_fence",
                )
                data = json.loads(json_text, parse_constant=_reject_nonfinite_json)
            else:
                data = _parse_json_object(raw)
            if not isinstance(data, dict):
                raise ValueError("reflection_object_required")
            if self.last_response_transport is not None:
                self.last_response_transport["json_status"] = "parsed"
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            self.last_status = "failed"
            self.last_error_type = "invalid_json_response"
            self.last_error_detail = _safe_reflection_error_detail(error, "invalid_json_response")
            if self.evidence_review_protocol:
                if self.last_response_transport is not None:
                    self.last_response_transport["json_status"] = "failed"
                self._record_proposal_outcomes(None, {}, review_status="not_evaluated")
            return persona

        raw_changes = data.get("changes")
        if (
            "changes" not in data
            or not isinstance(raw_changes, list)
            or not all(isinstance(item, dict) for item in raw_changes)
        ):
            self.last_status = "failed"
            self.last_error_type = "invalid_json_response"
            self.last_error_detail = "changes_array_required"
            if self.evidence_review_protocol:
                self._record_proposal_outcomes(None, {}, review_status="not_evaluated")
            return persona

        now = self._now().isoformat(timespec="seconds")
        reviews: list[PersonEvidenceReview] = []
        if self.evidence_review_protocol:
            raw_reviews = data.get("reviews")
            try:
                data, reviews = self._reviewed_changes(data, persona, ctx, now)
            except (ValueError, TypeError, KeyError) as error:
                self.last_status = "failed"
                self.last_error_type = "invalid_evidence_review_protocol"
                self.last_error_detail = _safe_reflection_error_detail(error, "invalid_evidence_review_protocol")
                self._record_proposal_outcomes(
                    raw_changes, {}, review_status="failed", error_detail=self.last_error_detail,
                )
                return persona
            self.last_belief_change_results = []
        changes = self._normalise_changes(data, persona, ctx)
        for change in changes:
            self._apply_change(
                persona, ctx.session_id, now, ctx, change
            )
        if self.evidence_review_protocol:
            applied_ids = {
                evidence.review_id
                for hypothesis in persona.person_hypotheses for evidence in hypothesis.evidence
                if evidence.session_id == ctx.session_id and evidence.review_id
            }
            for review in reviews:
                if review.id in applied_ids:
                    review.disposition = "applied"
            persona.evidence_review_ledger.extend(reviews)
            persona.evidence_review_ledger = persona.evidence_review_ledger[-_REVIEW_LEDGER_LIMIT:]
            self.last_review_summary = {
                "protocol_version": EVIDENCE_REVIEW_PROTOCOL_VERSION,
                "prompt_contract_version": EVIDENCE_REVIEW_PROMPT_CONTRACT_VERSION,
                "review_count": len(reviews), "applied_review_count": len(applied_ids),
                "dispositions": [review.disposition for review in reviews],
            }
            self._record_proposal_outcomes(
                raw_changes, {raw["id"]: review for raw, review in zip(raw_reviews, reviews)},
                review_status="completed",
                selected_review_ids={change["_review_record_id"] for change in changes},
            )

        state.sessions_observed += 1
        state.reflected_session_ids.append(ctx.session_id)
        state.reflected_session_ids = state.reflected_session_ids[-100:]
        state.last_reflected_at = now
        self.last_status = "ok"
        self.last_error_type = None
        return persona

    # 兼容原有调用名；语义已经从“诊断”变为“更新系统的可推翻假设”。
    def diagnose(self, persona: PersonaProfile, ctx) -> PersonaProfile:
        return self.update(persona, ctx)

    def _body_json_response(self, messages: list[dict]) -> str | None:
        """At most one format correction, before the evidence transaction.

        The first body is never echoed to the model. Every complete() remains
        a separate metered call; no HTTP retry or ledger reset happens here.
        Evidence/lineage/semantic validation runs only after this method and
        can never trigger a second answer.
        """
        from ..policy.reflection_json import (
            parse_reflection_json, reflection_body_system_prompt,
            reflection_correction_issues, reflection_correction_message,
            reflection_shape_diagnostic,
        )
        original = [{"role": "system", "content": reflection_body_system_prompt()}, messages[1].copy()]
        request = original
        for index in range(2):
            attempt = {"product_step": "initial" if index == 0 else "format_correction",
                       "attempt_index": index, "status": "started"}
            self.last_format_attempts.append(attempt)
            try:
                raw = self.llm.complete(request)
            except Exception:
                attempt["status"] = "model_call_failed"
                raise
            # Content-independent call diagnostics; the transport ledger owns
            # actual cost/unknown reservations and retains each physical call.
            usage = getattr(self.llm, "last_usage", None)
            if type(usage) is dict:
                attempt["usage"] = {key: value for key in ("prompt_tokens", "completion_tokens", "cost")
                                    if type(value := usage.get(key)) in (int, float)
                                    and math.isfinite(value) and value >= 0}
            body = raw
            try:
                body, _ = _unwrap_review_response(raw)
                parse_reflection_json(body)
            except (ValueError, TypeError, RecursionError) as error:
                attempt.update(status="shape_failed", shape=reflection_shape_diagnostic(error))
                issues = reflection_correction_issues(error, body)
                if index or not issues:
                    return None
                request = [*original, reflection_correction_message(issues)]
                self.last_format_correction_count = 1
                continue
            attempt["status"] = "shape_valid"
            return raw
        return None

    def _record_proposal_outcomes(
        self, raw_changes: list[dict] | None, reviews_by_local_id: dict,
        *, review_status: str, error_detail: str | None = None,
        selected_review_ids: set[str] | None = None,
    ) -> None:
        """Record bounded, content-free application diagnostics, not truth scores.

        A valid review can defer a proposal. A failed transaction rejects every
        submitted proposal, including ones preceding an invalid context review.
        No quote, claim, unvalidated identifier or provider error is copied here.
        """
        results = []
        for index, raw in enumerate((raw_changes or [])[:3]):
            relation = raw.get("relation")
            result = {
                "proposal_index": index,
                "proposed_relation": relation if isinstance(relation, str) and relation in _RELATIONS else None,
                "review_status": review_status,
                "status": "rejected", "reason": error_detail,
                "scope_basis": "not_evaluated", "contract_warning": None,
            }
            if review_status == "completed":
                review = reviews_by_local_id[raw["review_id"]]
                result["scope_basis"] = (
                    "new_atomic_conditions" if review.hypothesis_id is None else "existing_claim_scope"
                )
                result["reason"] = review.disposition
                if review.disposition == "applied":
                    result["status"] = "applied"
                elif review.disposition == "eligible_not_applied":
                    selected = review.id in (selected_review_ids or set())
                    result["status"] = "rejected" if selected else "deferred"
                    result["reason"] = (
                        "validated_change_not_applied" if selected else "not_selected_for_application"
                    )
                else:
                    result["status"] = "deferred"
                    if review.disposition != "same_claim_counter_takes_priority":
                        result["contract_warning"] = "ineligible_review_has_change"
            results.append(result)
        self.last_proposal_results = results
        counts = {status: sum(row["status"] == status for row in results)
                  for status in ("applied", "deferred", "rejected")}
        if review_status == "failed" and raw_changes is not None:
            # The entire invalid transaction is rejected, even when its
            # excessive proposal count exceeds the bounded diagnostic rows.
            counts["rejected"] = len(raw_changes)
        statuses = {row["status"] for row in results}
        status = (
            "not_evaluated" if raw_changes is None else
            "rejected" if review_status == "failed" else
            "no_proposal" if not results else
            next(iter(statuses)) if len(statuses) == 1 else "mixed"
        )
        self.last_application_summary = {
            "prompt_contract_version": EVIDENCE_REVIEW_PROMPT_CONTRACT_VERSION,
            "review_status": review_status, "status": status,
            "proposal_count": len(raw_changes) if raw_changes is not None else None,
            "proposal_results_count": len(results),
            "omitted_proposal_results_count": max(0, len(raw_changes or []) - len(results)),
            **{f"{key}_count": value for key, value in counts.items()},
        }

    def _reviewed_changes(self, data: dict, persona: PersonaProfile, ctx, now: str) -> tuple[dict, list[PersonEvidenceReview]]:
        """Validate 2.0 before mutating either ledger; relation truth is still reviewable.

        An exact quote proves provenance, not speaker ownership or semantic
        entailment. Those declared comparisons remain visible to independent
        assessment; this code enforces their consequences, not a keyword oracle.
        """
        if set(data) != {"protocol_version", "reviews", "changes"} or data["protocol_version"] != EVIDENCE_REVIEW_PROTOCOL_VERSION:
            raise ValueError("review_protocol_version_required")
        raw_reviews = data["reviews"]
        if not isinstance(raw_reviews, list) or len(raw_reviews) > 6 or len(data["changes"]) > 3:
            raise ValueError("review_or_change_bound_exceeded")
        fields = {
            "id", "hypothesis_id", "claim_quote", "scope_quote", "applicability", "relation",
            "evidence_basis", "event_kind", "prior_event_id", "evidence", "event_evidence",
            "reason", "event_summary",
        }
        known = {h.id: h for h in persona.person_hypotheses}
        known_events = {review.event_id for review in persona.evidence_review_ledger}
        known_events.update(e.event_id for h in persona.person_hypotheses for e in h.evidence if e.event_id)
        # Identity signatures use quoted event context, never an invented model
        # summary. A same signature cannot become a new event by changing its id.
        signatures = {
            _event_signature([q.quote for q in review.event_evidence]): review.event_id
            for review in persona.evidence_review_ledger if review.event_evidence
        }
        signatures.update({e.event_signature: e.event_id for h in persona.person_hypotheses
                           for e in h.evidence if e.event_signature and e.event_id})
        reviews: list[PersonEvidenceReview] = []
        by_local_id = {}
        for raw in raw_reviews:
            if not isinstance(raw, dict) or set(raw) != fields:
                raise ValueError("invalid_review_fields")
            local_id = raw["id"]
            if not isinstance(local_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", local_id) or local_id in by_local_id:
                raise ValueError("duplicate_or_invalid_review_id")
            hid = raw["hypothesis_id"]
            if hid is None:
                if raw["claim_quote"] is not None or raw["scope_quote"] is not None:
                    raise ValueError("new_review_has_unbound_prior_text")
            elif not isinstance(hid, str) or hid not in known or (
                raw["claim_quote"] != known[hid].claim or raw["scope_quote"] != (known[hid].scope or "")
            ):
                raise ValueError("review_prior_identity_mismatch")
            if raw["applicability"] not in {"within_scope", "outside_scope", "unknown"}:
                raise ValueError("invalid_scope_relation")
            if raw["relation"] not in {"support", "counter", "neutral", "unknown"}:
                raise ValueError("invalid_review_relation")
            if raw["evidence_basis"] not in {"stated_preference", "observed_response", "repeated_pattern", "context_only"}:
                raise ValueError("invalid_evidence_basis")
            if raw["event_kind"] not in {"new_event", "same_event", "retrospective", "unknown"}:
                raise ValueError("invalid_event_kind")
            prior_event = raw["prior_event_id"]
            if prior_event is not None and (not isinstance(prior_event, str) or prior_event not in known_events):
                raise ValueError("unknown_prior_event_reference")
            if raw["event_kind"] == "same_event" and prior_event is None:
                raise ValueError("same_event_requires_known_reference")
            if raw["event_kind"] == "new_event" and prior_event is not None:
                raise ValueError("new_event_cannot_reference_old_event")
            if any(not isinstance(raw[k], str) or not raw[k].strip() or len(raw[k]) > 240 for k in ("reason", "event_summary")):
                raise ValueError("invalid_bounded_review_explanation")
            evidence = _review_quotes(raw["evidence"], ctx)
            event_evidence = _review_quotes(raw["event_evidence"], ctx)
            signature = _event_signature([q.quote for q in event_evidence])
            matched_event = signatures.get(signature)
            if prior_event is not None and matched_event is not None and prior_event != matched_event:
                raise ValueError("event_reference_disagrees_with_known_quote_signature")
            event_id = prior_event or matched_event or "evt_" + hashlib.sha256(
                (persona.user_id + ":" + signature).encode("utf-8")
            ).hexdigest()[:24]
            disposition = "eligible_not_applied"
            if raw["applicability"] != "within_scope":
                disposition = "outside_scope_no_update" if raw["applicability"] == "outside_scope" else "unknown_scope_no_update"
            elif raw["relation"] not in {"support", "counter"} or raw["evidence_basis"] == "context_only":
                disposition = "context_only_no_update"
            elif raw["event_kind"] == "unknown":
                disposition = "event_identity_unknown_no_update"
            elif raw["event_kind"] == "same_event" or prior_event is not None or matched_event is not None:
                disposition = "same_event_no_fresh_evidence"
            elif raw["relation"] == "counter" and hid is None:
                raise ValueError("counter_requires_existing_hypothesis")
            prior_support = known[hid].supporting_events() if hid in known else set()
            independent_events = len(prior_support | {event_id})
            review = PersonEvidenceReview(
                id="rev_" + hashlib.sha256((ctx.session_id + ":" + local_id).encode("utf-8")).hexdigest()[:24],
                at=now, session_id=ctx.session_id, hypothesis_id=hid,
                claim_quote=raw["claim_quote"], scope_quote=raw["scope_quote"],
                applicability=raw["applicability"], relation=raw["relation"],
                evidence_basis=raw["evidence_basis"], event_kind=raw["event_kind"],
                event_id=event_id, prior_event_id=prior_event or matched_event,
                event_signature=signature,
                evidence=evidence, event_evidence=event_evidence,
                reason=raw["reason"], event_summary=raw["event_summary"], disposition=disposition,
                independent_event_count=independent_events,
                pattern_evidence_status=(
                    "not_pattern" if raw["evidence_basis"] != "repeated_pattern"
                    else "provisional_single_event" if independent_events < 2
                    else "multiple_independent_events_not_proof_of_stable_trait"
                ),
            )
            reviews.append(review)
            by_local_id[local_id] = review
        counter_targets = {r.hypothesis_id for r in reviews
                           if r.relation == "counter" and r.disposition == "eligible_not_applied"}
        prepared = []
        change_fields = {
            "review_id", "id", "framework", "dimension", "claim", "scope", "scope_tags", "relation",
            "lineage", "atomic_claim", "target_value", "confidence", "evidence",
        }
        used_reviews = set()
        for raw in data["changes"]:
            if set(raw) != change_fields or raw.get("review_id") not in by_local_id:
                raise ValueError("change_requires_exact_review_binding")
            local_id = raw["review_id"]
            if local_id in used_reviews:
                raise ValueError("one_review_cannot_duplicate_changes")
            used_reviews.add(local_id)
            review = by_local_id[local_id]
            atomic = _atomic_claim(raw["atomic_claim"])
            if atomic.evidence_basis != review.evidence_basis:
                raise ValueError("change_evidence_basis_mismatch")
            quotes = _review_quotes(raw["evidence"], ctx)
            if not {(q.turn_index, q.quote) for q in quotes} <= {(q.turn_index, q.quote) for q in review.evidence}:
                raise ValueError("change_quote_not_in_bound_review")
            if (review.hypothesis_id is not None and review.relation == "support"
                    and raw["relation"] != "support"):
                raise ValueError("existing_support_requires_support_change")
            expected_relations = {"new", "support"} if review.relation == "support" else {"counter", "retire"}
            if raw["relation"] not in expected_relations:
                raise ValueError("change_relation_disagrees_with_review")
            if type(raw["confidence"]) not in (int, float) or not math.isfinite(raw["confidence"]) or not 0 <= raw["confidence"] <= 1:
                raise ValueError("invalid_change_confidence")
            tags = raw["scope_tags"]
            if (not isinstance(tags, list) or not 1 <= len(tags) <= 8
                    or any(not isinstance(tag, str) or tag not in _SCOPE_TAGS for tag in tags)
                    or len(set(tags)) != len(tags) or ("general" in tags and len(tags) > 1)):
                raise ValueError("invalid_explicit_scope_tags")
            value = raw["target_value"]
            if value is not None and (type(value) not in (int, float) or not math.isfinite(value)
                                      or not 0 <= value <= 1 or raw["framework"] not in {"big_five", "disc"}):
                raise ValueError("invalid_or_inapplicable_person_scale")
            item = dict(raw)
            if review.hypothesis_id is None:
                if raw["id"] not in {None, ""} or raw["lineage"] is not None or raw["relation"] != "new":
                    raise ValueError("new_claim_requires_unbound_identity")
                item["claim"] = _atomic_statement(atomic)
                item["scope"] = atomic.conditions
                if any(h.claim == item["claim"] and h.scope == item["scope"] for h in known.values()):
                    raise ValueError("existing_claim_requires_explicit_review_identity")
            else:
                old = known[review.hypothesis_id]
                lineage = {"id": old.id, "claim_quote": old.claim, "scope_quote": old.scope or "",
                           "mechanism_relation": "same", "scope_relation": "within"}
                if raw["id"] != old.id or raw["claim"] != old.claim or raw["lineage"] != lineage:
                    raise ValueError("change_existing_claim_identity_mismatch")
            if not self._valid_change_shape(item):
                raise ValueError("invalid_reviewed_change_shape")
            if review.disposition != "eligible_not_applied":
                continue
            if review.hypothesis_id in counter_targets and review.relation == "support":
                review.disposition = "same_claim_counter_takes_priority"
                continue
            item.update({
                "_review_validated": True, "_review_record_id": review.id,
                "_review_event_id": review.event_id, "_review_event_kind": review.event_kind,
                "_review_event_signature": review.event_signature,
                "_review_evidence_basis": review.evidence_basis,
                "_atomic_claim": atomic.model_dump(),
            })
            prepared.append(item)
        return {"changes": prepared}, reviews

    def _normalise_changes(
        self,
        data: dict,
        persona: PersonaProfile,
        ctx,
    ) -> list[dict]:
        """Enforce one highest-value update even if a model over-produces.

        Existing lifecycle corrections outrank a new adjacent label.  Within the
        same class, model order is retained because the prompt requires strongest
        first. Validate shape, requested identity and current-session quote
        grounding before ranking: an unsupported "counter" must not suppress a
        later grounded update merely because lifecycle corrections rank first.
        ``_apply_change`` repeats grounding as a defensive boundary.
        """
        raw = data.get("changes")
        if not isinstance(raw, list):
            return []
        changes = [
            item for item in raw
            if isinstance(item, dict) and self._valid_change_shape(item)
        ]
        if not changes:
            return []

        ranked: list[tuple[int, int, dict]] = []
        for index, item in enumerate(changes):
            original = item
            item = self._resolve_explicit_lineage(item, persona.person_hypotheses)
            if item is None:
                continue
            framework = str(item.get("framework") or "").strip().lower()
            dimension = _compact(item.get("dimension"), 60).lower() or None
            relation = str(item.get("relation") or "").strip().lower()
            if relation == "no_change" or not _normalise_evidence(
                item.get("evidence"), ctx
            ):
                continue
            requested_id = _compact(item.get("id"), 80)
            if requested_id and not any(
                hypothesis.id == requested_id
                and hypothesis.framework == framework
                and hypothesis.dimension == dimension
                for hypothesis in persona.person_hypotheses
            ):
                continue
            match = None if self.evidence_review_protocol and not requested_id else self._find_hypothesis(
                persona.person_hypotheses,
                item.get("id"),
                framework,
                dimension,
                _compact(item.get("claim")),
                _compact(item.get("scope")) or None,
                _normalise_scope_tags(item.get("scope_tags")),
                relation,
            )
            if match is None and relation in {"counter", "retire"}:
                continue
            lifecycle_priority = 2 if (
                match is not None
                and relation in {"counter", "retire"}
            ) else 1 if match is not None else 0
            # Keep the actual proposal for _apply_change to independently resolve
            # again, including its original classification for the audit marker.
            ranked.append((lifecycle_priority, -index, original))
        if not ranked:
            return []
        return [max(ranked, key=lambda row: (row[0], row[1]))[2]]

    @staticmethod
    def _valid_change_shape(change: dict) -> bool:
        framework = str(change.get("framework") or "").strip().lower()
        relation = str(change.get("relation") or "").strip().lower()
        dimension = _compact(change.get("dimension"), 60).lower() or None
        return (
            framework in _FRAMEWORKS
            and relation in _RELATIONS
            and dimension in _FRAMEWORK_DIMENSIONS[framework]
            and bool(_compact(change.get("claim")))
            and isinstance(change.get("evidence"), list)
        )

    @staticmethod
    def _resolve_explicit_lineage(
        change: dict,
        hypotheses: Iterable[PersonHypothesis],
    ) -> dict | None:
        """Resolve an explicit claim comparison, never infer across labels.

        The offline model must name one existing identity, quote its complete
        claim/scope and declare the same mechanism within that scope. These
        checks validate the *reference*, not the semantic truth of the model's
        declaration; current quoted evidence is still required separately. A
        partial/contradictory comparison fails closed instead of creating a new
        label. Older payloads without this optional contract keep their existing
        same-dimension matching behavior.
        """
        resolved = {
            key: value for key, value in change.items()
            if not key.startswith("_lineage_")
        }
        lineage = change.get("lineage")
        if lineage is None:
            return resolved
        if not isinstance(lineage, dict):
            return None
        if (
            change.get("relation") not in {"support", "counter", "retire"}
            or lineage.get("mechanism_relation") != "same"
            or lineage.get("scope_relation") != "within"
        ):
            return None
        identity = lineage.get("id")
        if not isinstance(identity, str) or not identity:
            return None
        matches = [item for item in hypotheses if item.id == identity]
        if len(matches) != 1:
            return None
        existing = matches[0]
        if (
            not isinstance(lineage.get("claim_quote"), str)
            or not isinstance(lineage.get("scope_quote"), str)
            or lineage["claim_quote"] != existing.claim
            or lineage["scope_quote"] != (existing.scope or "")
            or _compact(change.get("id"), 80) not in {"", existing.id}
        ):
            return None
        framework = str(change.get("framework") or "").strip().lower()
        dimension = _compact(change.get("dimension"), 60).lower()
        resolved.update({
            "id": existing.id,
            "framework": existing.framework,
            "dimension": existing.dimension,
            "claim": existing.claim,
            "_lineage_source": (
                f"conversation_quote:lineage:{framework}/{dimension}"
            ),
            "_lineage_reclassified": (
                framework != existing.framework or dimension != existing.dimension
            ),
        })
        return resolved

    def _apply_change(
        self,
        persona: PersonaProfile,
        session_id: str,
        now: str,
        ctx,
        change: dict,
    ) -> None:
        if self.evidence_review_protocol and change.get("_review_validated") is not True:
            return
        change = self._resolve_explicit_lineage(change, persona.person_hypotheses)
        if change is None:
            return
        framework = str(change.get("framework") or "").strip().lower()
        relation = str(change.get("relation") or "").strip().lower()
        if framework not in _FRAMEWORKS or relation not in _RELATIONS:
            return
        if relation == "no_change":
            return
        dimension = _compact(change.get("dimension"), 60).lower() or None
        if dimension not in _FRAMEWORK_DIMENSIONS[framework]:
            return
        claim = _compact(change.get("claim"))
        scope = _compact(change.get("scope")) or None
        scope_tags = _normalise_scope_tags(change.get("scope_tags"))
        if not claim:
            return
        evidence_rows = change.get("evidence") or []
        evidence = _normalise_evidence(evidence_rows, ctx)
        if relation in {"new", "support", "counter", "retire"} and not evidence:
            return

        requested_id = _compact(change.get("id"), 80)
        if requested_id:
            requested = next(
                (
                    item for item in persona.person_hypotheses
                    if item.id == requested_id
                ),
                None,
            )
            if (
                requested is None
                or requested.framework != framework
                or requested.dimension != dimension
            ):
                return

        hypothesis = None if self.evidence_review_protocol and not requested_id else self._find_hypothesis(
            persona.person_hypotheses,
            requested_id,
            framework,
            dimension,
            claim,
            scope,
            scope_tags,
            relation,
        )
        if self.evidence_review_protocol and hypothesis is not None and any(
            item.event_id == change.get("_review_event_id") for item in hypothesis.evidence
        ):
            # Same event is not another independent vote, even if a caller
            # bypasses the normal reviewed-change preparation path.
            return
        before = belief_state(hypothesis)
        if hypothesis is None:
            if relation in {"counter", "retire"}:
                return
            hypothesis = PersonHypothesis(
                id=_hypothesis_id(framework, dimension, claim, scope),
                framework=framework,
                dimension=dimension,
                claim=claim,
                scope=scope,
                scope_tags=scope_tags,
                target_value=(
                    _clamp(change.get("target_value"))
                    if isinstance(change.get("target_value"), (int, float))
                    else None
                ),
                confidence=0.0,
                status="candidate",
                first_observed_at=now,
                atomic_claim=(AtomicPersonClaim(**change["_atomic_claim"])
                              if self.evidence_review_protocol else None),
            )
            hypothesis.status_history.append(PersonHypothesisTransition(
                at=now,
                session_id=session_id,
                from_status="none",
                to_status="candidate",
                reason="首次出现带轮次出处的可观察证据",
            ))
            persona.person_hypotheses.append(hypothesis)

        reported_confidence = _clamp(change.get("confidence"))
        restarting = (
            hypothesis.status == "retired" and relation in {"new", "support"}
        )
        if restarting:
            hypothesis.activation_epoch += 1
            hypothesis.confidence = 0.0
        if not change.get("_lineage_reclassified") and relation in {
            "new", "support"
        } and isinstance(
            change.get("target_value"), (int, float)
        ):
            target = _clamp(change["target_value"])
            hypothesis.target_value = (
                target if hypothesis.target_value is None
                else hypothesis.target_value * 0.7 + target * 0.3
            )
        if (
            not change.get("_lineage_source")
            and relation in {"new", "support"}
            and scope and not hypothesis.scope
        ):
            hypothesis.scope = scope
        if (
            not change.get("_lineage_source")
            and relation in {"new", "support"} and scope_tags
        ):
            hypothesis.scope_tags = _normalise_scope_tags([
                *hypothesis.scope_tags,
                *scope_tags,
            ])[:8]

        evidence_relation = (
            "counter" if relation in {"counter", "retire"} else "support"
        )
        evidence_recorded_count = 0
        for row in evidence[:5]:
            record = PersonHypothesisEvidence(
                at=now,
                session_id=session_id,
                relation=evidence_relation,
                summary=row["summary"],
                turn_indices=row["turn_indices"],
                source=_compact(
                    change.get("_lineage_source")
                    or row.get("source") or "conversation", 80
                ),
                scope=scope,
                scope_tags=scope_tags,
                activation_epoch=hypothesis.activation_epoch,
                **({"event_id": change["_review_event_id"], "review_id": change["_review_record_id"],
                    "evidence_basis": change["_review_evidence_basis"], "event_kind": change["_review_event_kind"],
                    "event_signature": change["_review_event_signature"]}
                   if self.evidence_review_protocol else {}),
            )
            duplicate = any(
                item.session_id == record.session_id
                and item.activation_epoch == record.activation_epoch
                and item.relation == record.relation
                and item.summary == record.summary
                and item.turn_indices == record.turn_indices
                for item in hypothesis.evidence
            )
            if not duplicate:
                hypothesis.evidence.append(record)
                evidence_recorded_count += 1
        hypothesis.evidence = hypothesis.evidence[-40:]
        if not evidence_recorded_count:
            return
        hypothesis.confidence, effective_relation, change_reason = update_confidence(
            hypothesis.confidence, relation, reported_confidence, evidence_recorded_count,
        )
        if relation in {"new", "support"}:
            hypothesis.last_confirmed_at = now
        previous_status = hypothesis.status
        self._refresh_status(hypothesis)
        if hypothesis.status != previous_status:
            hypothesis.status_history.append(PersonHypothesisTransition(
                at=now,
                session_id=session_id,
                from_status=previous_status,
                to_status=hypothesis.status,
                reason=self._transition_reason(hypothesis),
            ))
            hypothesis.status_history = hypothesis.status_history[-30:]
        if self.evidence_review_protocol and self.last_belief_change_results is not None:
            after = belief_state(hypothesis)
            self.last_belief_change_results.append({
                "confidence_policy_version": CONFIDENCE_UPDATE_POLICY_VERSION,
                "review_id": change["_review_record_id"], "hypothesis_id": hypothesis.id,
                "proposed_relation": relation, "effective_relation": effective_relation,
                "evidence_recorded_count": evidence_recorded_count,
                "model_evidence_strength": reported_confidence,
                **{f"{key}_before": value for key, value in before.items()
                   if key != "claim_scope_fingerprint"},
                **{f"{key}_after": value for key, value in after.items()
                   if key != "claim_scope_fingerprint"},
                "claim_scope_before_fingerprint": before["claim_scope_fingerprint"],
                "claim_scope_after_fingerprint": after["claim_scope_fingerprint"],
                "belief_changed": before != after,
                "actually_retired": before["status"] != "retired" and after["status"] == "retired",
                "reactivated": before["status"] == "retired" and after["status"] != "retired",
                "reason": change_reason,
            })

    @staticmethod
    def _find_hypothesis(
        hypotheses: Iterable[PersonHypothesis],
        requested_id: Any,
        framework: str,
        dimension: str | None,
        claim: str,
        scope: str | None,
        scope_tags: Iterable[str] = (),
        relation: str = "support",
    ) -> PersonHypothesis | None:
        pool = list(hypotheses)
        requested = _compact(requested_id, 80)
        if requested:
            for item in pool:
                if item.id == requested:
                    return item
        eligible = [
            item for item in pool
            if item.framework == framework
            and item.dimension == dimension
            and (item.status != "retired" or relation in {"new", "support"})
        ]
        # Prefer an existing live lineage if legacy profiles already contain
        # duplicates. A retired exact match remains available for fresh support.
        eligible.sort(key=lambda item: item.status == "retired")
        for item in eligible:
            if (
                _compact(item.claim).lower() == _compact(claim).lower()
                and _compact(item.scope).lower() == _compact(scope).lower()
            ):
                return item

        incoming_tags = _normalise_scope_tags(list(scope_tags))
        close: list[tuple[float, PersonHypothesis]] = []
        for item in eligible:
            tag_similarity = _scope_tag_similarity(
                item.scope_tags, incoming_tags
            )
            claim_similarity = _text_similarity(item.claim, claim)
            scope_similarity = _text_similarity(item.scope, scope)
            # Both structured context and wording must agree.  This deliberately
            # leaves distant paraphrases separate instead of guessing identity.
            if (
                tag_similarity >= 0.6
                and claim_similarity >= 0.34
                and (scope_similarity >= 0.3 or claim_similarity >= 0.58)
            ) or (
                not item.scope_tags
                and not incoming_tags
                and claim_similarity >= 0.62
                and (
                    (not item.scope and not scope)
                    or scope_similarity >= 0.45
                )
            ):
                close.append((
                    claim_similarity + scope_similarity + tag_similarity,
                    item,
                ))
        if close:
            close.sort(key=lambda row: row[0], reverse=True)
            # If two hypotheses are similarly plausible, omission of the id is
            # ambiguous and must fail closed rather than merge different meanings.
            if len(close) == 1 or close[0][0] - close[1][0] >= 0.2:
                return close[0][1]

        # Counterevidence is expected to use opposite wording, so lexical
        # similarity is not meaningful.  Infer a missing id only when exactly one
        # active hypothesis has the same framework/dimension and strongly
        # overlapping context; multiple candidates remain ambiguous.
        if relation in {"counter", "retire"}:
            scoped = [
                item for item in eligible
                if (
                    _scope_tag_similarity(item.scope_tags, incoming_tags) >= 0.6
                    or (
                        not item.scope_tags
                        and not incoming_tags
                        and _text_similarity(item.scope, scope) >= 0.55
                    )
                )
            ]
            if len(scoped) == 1:
                return scoped[0]
        return None

    @staticmethod
    def _refresh_status(hypothesis: PersonHypothesis) -> None:
        evidence = hypothesis.activation_evidence()
        strict_events = any(item.event_id is not None for item in evidence)
        support_sessions = len({
            (item.event_id or "legacy_session:" + item.session_id) if strict_events else item.session_id
            for item in evidence if item.relation == "support"
        })
        support_days = len({
            item.at[:10] for item in evidence
            if item.relation == "support" and len(item.at) >= 10
        })
        counter_sessions = len({
            (item.event_id or "legacy_session:" + item.session_id) if strict_events else item.session_id
            for item in evidence if item.relation == "counter"
        })
        support_rows = len({
            (item.session_id, turn_index)
            for item in evidence if item.relation == "support"
            for turn_index in item.turn_indices
        })
        if hypothesis.confidence <= 0.15:
            hypothesis.status = "retired"
        elif (
            support_sessions >= 5
            and support_days >= 3
            and support_rows >= 7
            and hypothesis.confidence >= 0.75
            and counter_sessions <= 1
        ):
            hypothesis.status = "stable"
        elif (
            support_sessions >= 2
            and support_rows >= 3
            and hypothesis.confidence >= 0.5
        ):
            hypothesis.status = "working"
        else:
            hypothesis.status = "candidate"

    @staticmethod
    def _transition_reason(hypothesis: PersonHypothesis) -> str:
        evidence = hypothesis.activation_evidence()
        support_rows = len({
            (item.session_id, turn_index)
            for item in evidence if item.relation == "support"
            for turn_index in item.turn_indices
        })
        current_support_sessions = len({
            item.session_id for item in evidence if item.relation == "support"
        })
        return (
            f"激活周期 {hypothesis.activation_epoch}；"
            f"本周期支持会话 {current_support_sessions}；"
            f"支持会话 {len(hypothesis.supporting_sessions())}；"
            f"覆盖自然日 {len(hypothesis.supporting_days())}；"
            f"支持轮次证据 {support_rows}；"
            f"反证会话 {len(hypothesis.counter_sessions())}；"
            f"置信 {hypothesis.confidence:.2f}"
        )

def _event_signature(quotes: list[str]) -> str:
    return hashlib.sha256(json.dumps(sorted(set(quotes)), ensure_ascii=False).encode("utf-8")).hexdigest()


def _reject_nonfinite_json(value: str):
    raise ValueError("nonfinite_json_number")


def _review_quotes(rows: Any, ctx) -> list[EvidenceReviewQuote]:
    if not isinstance(rows, list) or not 1 <= len(rows) <= 3:
        raise ValueError("bounded_current_quotes_required")
    excluded = _sensitive_memory_excluded_turns(ctx)
    result = []
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"turn_index", "evidence_quote"}:
            raise ValueError("invalid_review_quote_shape")
        index, quote = row["turn_index"], row["evidence_quote"]
        if (type(index) is not int or not 0 <= index < len(ctx.turns) or index in excluded
                or not isinstance(quote, str) or not 2 <= len(quote) <= 240
                or quote not in ctx.turns[index].user or (index, quote) in seen):
            raise ValueError("review_quote_not_grounded_in_current_user")
        seen.add((index, quote))
        result.append(EvidenceReviewQuote(turn_index=index, quote=quote))
    return result


def _atomic_claim(raw: Any) -> AtomicPersonClaim:
    if not isinstance(raw, dict) or set(raw) != {"aspect", "conditions", "tendency", "outcome", "evidence_basis"}:
        raise ValueError("one_atomic_claim_required")
    if any(not isinstance(raw[key], str) or not raw[key].strip() or len(raw[key]) > 100
           for key in ("aspect", "conditions", "tendency")):
        raise ValueError("invalid_atomic_claim_text")
    if raw["evidence_basis"] not in {"stated_preference", "observed_response", "repeated_pattern"}:
        raise ValueError("context_or_authorization_is_not_person_claim")
    outcome = raw["outcome"]
    if outcome is not None and (not isinstance(outcome, str) or not outcome.strip() or len(outcome) > 100):
        raise ValueError("invalid_atomic_outcome")
    if raw["evidence_basis"] == "stated_preference" and outcome is not None:
        raise ValueError("stated_preference_must_not_invent_causal_outcome")
    if raw["evidence_basis"] == "observed_response" and outcome is None:
        raise ValueError("observed_response_requires_reported_outcome")
    atomic = AtomicPersonClaim(**raw)
    _atomic_statement(atomic)
    return atomic


def _atomic_statement(atomic: AtomicPersonClaim) -> str:
    value = f"在{atomic.conditions}的情境下，可能{atomic.tendency}"
    if atomic.outcome:
        value += f"；本人描述的伴随反应是{atomic.outcome}"
    if len(value) > 180:
        raise ValueError("atomic_claim_exceeds_retained_claim_bound")
    return value


def _normalise_evidence(rows: Any, ctx) -> list[dict]:
    excluded = _sensitive_memory_excluded_turns(ctx)
    by_turn: dict[int, dict] = {}
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        turn_index = row.get("turn_index")
        quote = str(row.get("evidence_quote") or "").strip()
        if (
            isinstance(turn_index, bool)
            or not isinstance(turn_index, int)
            or not 0 <= turn_index < len(ctx.turns)
            or turn_index in excluded
            or len(quote) < 2
            or quote not in ctx.turns[turn_index].user
        ):
            continue
        stored_quote = quote[:160]
        record = {
            "summary": f"用户原话：“{stored_quote}”",
            "turn_indices": [turn_index],
            "source": "conversation_quote",
        }
        # Several copied substrings from one utterance are one observation, not
        # several independent votes. Keep the longest grounded quote for audit.
        previous = by_turn.get(turn_index)
        if previous is None or len(record["summary"]) > len(previous["summary"]):
            by_turn[turn_index] = record
    return list(by_turn.values())


def _sensitive_memory_excluded_turns(ctx) -> set[int]:
    raw = ctx.working_memory.get("_sensitive_memory_excluded_turns") or []
    return {
        item for item in raw
        if isinstance(item, int) and not isinstance(item, bool) and item >= 0
    }


# 兼容公开导入；新代码建议使用更准确的 PersonaHypothesisUpdater。
PersonaDiagnoser = PersonaHypothesisUpdater
