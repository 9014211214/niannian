"""Persona + Episodic Memory schema(从旧 repo 移植,schema 保持一致)。

设计原则:
- demographics(PII)是 optional,不参与性格推理。
- DISC + Big Five + 依恋 + cognitive_profile 是性格主线。
- episodic_memory 每条带 referenceable 标记 —— Aria 可以「内化但不挑明」。
- aria_strategy_hints 是从 persona 推导出的回应策略(prompt 直接注入)。

生产侧由会后模型从长期对话更新,demo 阶段可用预写合成人物,schema 一致；
case_conceptualization 与在线 Formulation 已形成可反证、可跨会话累积的闭环。
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, field_validator, model_validator


LOCAL_DEMO_NOTIFICATION_SCOPE = "local_demo_care_notifications"
CARE_NOTIFICATION_CONSENT_SCOPES = frozenset({
    "care_plan_notifications",
    "care_and_safety_notifications",
})
SAFETY_NOTIFICATION_CONSENT_SCOPES = frozenset({
    "safety_support_notifications",
    "care_and_safety_notifications",
})
AD_HOC_NOTIFICATION_CONSENT_SCOPES = frozenset({
    "ad_hoc_contact_notifications",
    "care_and_safety_notifications",
})


class Demographics(BaseModel):
    """OPTIONAL —— 不进入性格推理路径。"""
    name: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    occupation: Optional[str] = None
    location: Optional[str] = None
    hometown: Optional[str] = None
    education: Optional[str] = None


class Background(BaseModel):
    life_stage: Optional[str] = None
    recent_events: List[str] = []
    relationship_status: Optional[str] = None
    social_circle: Optional[str] = None


class DISC(BaseModel):
    dominance: float = 0.5
    influence: float = 0.5
    steadiness: float = 0.5
    conscientiousness: float = 0.5


class BigFive(BaseModel):
    openness: float = 0.5
    conscientiousness: float = 0.5
    extraversion: float = 0.5
    agreeableness: float = 0.5
    neuroticism: float = 0.5


class CognitiveProfile(BaseModel):
    distortion_tendencies: List[str] = []
    intervention_receptiveness: float = 0.5


class CaseConceptualization(BaseModel):
    """CBT-informed 慢变量档案摘要。

    不是用户可见的诊断。它捕捉 Aria 在选择运行时回应策略前,
    应当先内化的长期模式。
    """
    core_patterns: List[str] = Field(default_factory=list)
    triggers: List[str] = Field(default_factory=list)
    automatic_thoughts_or_beliefs: List[str] = Field(default_factory=list)
    emotion_behavior_loop: Optional[str] = None
    maintenance_factors: List[str] = Field(default_factory=list)
    protective_factors: List[str] = Field(default_factory=list)


class BehavioralMarkers(BaseModel):
    speech_pace: Optional[str] = None
    utterance_length: Optional[str] = None
    pitch_pattern: Optional[str] = None
    filler_words: List[str] = []
    emoji_usage: Optional[str] = None
    speech_examples: List[str] = []


class EmotionalBaseline(BaseModel):
    default_mood: Optional[str] = None
    trigger_topics: List[str] = []
    recovery_speed: Optional[str] = None
    hidden_emotions: List[str] = []


class EpisodicMemoryEntry(BaseModel):
    """单条情景记忆。referenceable=true 时 Aria 可以主动引用。"""
    id: str
    timestamp: Optional[str] = None
    type: Optional[str] = None
    summary: str
    keywords: List[str] = []
    sentiment: Optional[str] = None
    referenceable: StrictBool = False
    # Optional lineage for new extractions. Legacy/imported memories keep
    # missing provenance; a timestamp or matching text is not a source ID.
    source_session_id: Optional[str] = None
    source_turn_index: Optional[StrictInt] = Field(default=None, ge=0)
    source_quote: Optional[str] = None
    # Full owner turn supplied by the engine, not by the model. Missing legacy
    # context remains readable but cannot authorize proactive reference.
    source_context: Optional[str] = None
    source_context_sha256: Optional[str] = None
    source_context_version: Optional[str] = None

    def can_reference(self) -> bool:
        from ..memory.referenceability import (
            must_not_reference, reference_context_is_valid,
        )
        return bool(
            self.referenceable is True
            and isinstance(self.source_session_id, str) and self.source_session_id.strip()
            and type(self.source_turn_index) is int and self.source_turn_index >= 0
            and reference_context_is_valid(
                self.source_context, self.source_context_sha256,
                self.source_context_version, self.source_quote,
            )
            and not must_not_reference(self.summary)
        )

    @field_validator("timestamp", mode="before")
    @classmethod
    def _coerce_timestamp(cls, v):
        # YAML 会把 2026-04-08 parse 成 datetime.date,统一转成字符串。
        if v is None:
            return None
        return str(v)


class AriaStrategyHints(BaseModel):
    """直接注入 prompt 的回应策略,structure 任意 dict 透传。"""
    pace: Optional[str] = None
    length: Optional[str] = None
    tone: Optional[str] = None
    praise_style: Optional[Dict[str, Any]] = None
    comfort_style: Optional[Dict[str, Any]] = None
    silence_tolerance: Optional[str] = None
    conflict_style: Optional[str] = None
    episodic_recall_style: Optional[str] = None
    emotional_handling: Optional[str] = None
    topic_caution: List[str] = []
    preferred_actions: List[str] = []
    # 离线复盘产物:基于本次/长期交互后验总结出的动态回应策略。
    tactics: List[str] = []


class Medication(BaseModel):
    name: str
    dosage: Optional[str] = None
    schedule: Optional[str] = None       # 如「每天一次,早餐后」(自由文本,人读)
    at: Optional[str] = None             # 可选结构化时点「HH:MM」—— 主动提醒的用药时间窗触发用


class EmergencyContact(BaseModel):
    contact_id: Optional[str] = None      # 档案内稳定标识；不能只靠“女儿”等显示名路由
    name: str
    relation: Optional[str] = None
    channel: Optional[str] = None        # phone / sms / ...
    recipient_ref: Optional[str] = None  # 外部照护系统中的不透明收件人引用
    consent_to_notify: bool = False      # 是否已同意被通知(隐私:默认 False,不偷偷上报)
    notification_consent_scope: Optional[str] = None
    notification_consent_source: Optional[str] = None
    notification_consented_at: Optional[str] = None
    notification_consent_revoked_at: Optional[str] = None

    def has_auditable_notification_consent(
        self,
        allowed_scopes: frozenset[str],
    ) -> bool:
        """只接受范围明确、来源可追溯且带时区时间的当前授权。"""
        if not self.consent_to_notify:
            return False
        if (self.notification_consent_scope or "").strip() not in allowed_scopes:
            return False
        if not (self.notification_consent_source or "").strip():
            return False
        raw_time = (self.notification_consented_at or "").strip()
        if not raw_time:
            return False
        try:
            consented_at = datetime.fromisoformat(
                raw_time.replace("Z", "+00:00")
            )
        except ValueError:
            return False
        if consented_at.utcoffset() is None:
            return False
        # 重新授权必须清掉 revoked_at；矛盾记录一律 fail closed。
        if (self.notification_consent_revoked_at or "").strip():
            return False
        return True


class AllianceRecord(BaseModel):
    """跨会话的关系账本(P2-#10)—— trust 不再每次会话重置 0.5。

    关系本身是长期复利的一部分:会话开始时播种进 EngineContext,结束时写回。
    """
    trust: float = 0.5
    rupture_count: int = 0               # 历史破裂次数(修复能力的输入,也是预警信号)


class DeliveryModeState(BaseModel):
    """交付模式滞回(P2-#9)—— 防阈值悬崖导致会话间交付翻转。

    换挡需连续 N 次会话给出一致的新证据;单次诊断波动不改变对他说话的方式。
    """
    committed: Optional[str] = None      # 当前生效的交付模式(None=尚未定挡)
    hypothesis_basis_fingerprint: Optional[str] = None
    candidate: Optional[str] = None     # 待确认的新挡位
    streak: int = 0                      # candidate 连续一致的会话数


class RiskEvent(BaseModel):
    """历史或模型分流记录；不是本人危险、人格或纵向变化的证明。

    note 只存证据摘要(截断),不存全文;整个账本随档案一起受
    PIPL 删除权与(可选的)静态加密保护。
    """
    at: str                              # ISO 时间
    tier: int
    kind: str                            # ideation / high_distress / low_mood ...
    note: Optional[str] = None


class SafetyScreeningRecord(BaseModel):
    """Operational keyword takeover audit; subject is deliberately unverified.

    This contains no raw utterance, permission or person/health inference and
    must not be consumed as ordinary memory, a person trait or a care consent.
    """
    model_config = ConfigDict(extra="forbid")
    at: str
    session_id: str = Field(min_length=1, max_length=128)
    turn_index: StrictInt = Field(ge=0)
    tier: StrictInt = Field(ge=2, le=3)
    kind: str = Field(min_length=1, max_length=128)
    source: Literal["keyword_backstop"] = "keyword_backstop"
    subject: Literal["unverified"] = "unverified"
    evidence_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class InterventionOutcomeRecord(BaseModel):
    """一次干预的最小结果账本；只存结构化结果，不保存原始音视频。"""

    at: str
    session_id: Optional[str] = None
    session_turn: int
    intervention_id: str
    intervention_label: str
    goal: str
    outcome: str                         # positive / neutral / negative
    signal: str
    note: Optional[str] = None


class BaselineStat(BaseModel):
    """一项个人时间基线的在线统计量，不保存原始音视频或逐帧数据。"""

    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    updated_at: Optional[str] = None


class LongitudinalBaseline(BaseModel):
    """语速、停顿、音量和行为参与等指标的逐人基线。"""

    signals: Dict[str, BaselineStat] = Field(default_factory=dict)


class PersonHypothesisEvidence(BaseModel):
    """长期人物假设的一条支持或反证；只保存摘要与来源，不保存原始媒体。"""

    at: str
    session_id: str
    relation: str                         # support / counter
    summary: str
    turn_indices: List[int] = Field(default_factory=list)
    source: str = "conversation"
    scope: Optional[str] = None
    scope_tags: List[str] = Field(default_factory=list)
    # A retired hypothesis keeps its audit trail, but a renewed candidate must
    # earn activation from fresh evidence rather than reuse retired support.
    activation_epoch: int = 0
    # Protocol 2.0 links a quoted occurrence to a reviewed event. Legacy rows
    # have no event id and remain readable; a replay is never fresh evidence.
    event_id: Optional[str] = None
    review_id: Optional[str] = None
    evidence_basis: Optional[str] = None
    event_kind: Optional[str] = None
    event_signature: Optional[str] = None


class AtomicPersonClaim(BaseModel):
    """One contextual person tendency, never an external-action permission."""

    aspect: str
    conditions: str
    tendency: str
    outcome: Optional[str] = None
    evidence_basis: str


class EvidenceReviewQuote(BaseModel):
    turn_index: int
    quote: str


class PersonEvidenceReview(BaseModel):
    """Bounded quote-grounded observation, including decisions NOT to update.

    Relation, event summary and reason are model assessments, not user facts.
    Keeping an unknown/rejected observation does not activate a person claim.
    """

    id: str
    protocol_version: str = "2.0"
    at: str
    session_id: str
    hypothesis_id: Optional[str] = None
    claim_quote: Optional[str] = None
    scope_quote: Optional[str] = None
    applicability: str
    relation: str
    evidence_basis: str
    event_kind: str
    event_id: str
    prior_event_id: Optional[str] = None
    event_signature: Optional[str] = None
    evidence: List[EvidenceReviewQuote] = Field(default_factory=list)
    event_evidence: List[EvidenceReviewQuote] = Field(default_factory=list)
    reason: str
    event_summary: str
    disposition: str = "not_applied"
    independent_event_count: int = 0
    pattern_evidence_status: str = "not_pattern"
    assessment_scope: str = "model_review_not_user_testimony_or_authorization"


class PersonHypothesisTransition(BaseModel):
    """假设生命周期迁移，证明系统的认识如何形成或被推翻。"""

    at: str
    session_id: str
    from_status: str
    to_status: str
    reason: str


class PersonHypothesis(BaseModel):
    """Aria 对这个人的可推翻长期假设，而不是给这个人下的永久标签。"""

    id: str
    framework: str                       # big_five / disc / attachment / cbt / response
    dimension: Optional[str] = None
    claim: str
    scope: Optional[str] = None           # 假设在哪类情境下成立
    scope_tags: List[str] = Field(default_factory=list)
    # response 假设使用结构化场景标签，防止“在陌生群体中不喜欢被催”
    # 被错误泛化到所有对话。自然语言 scope 继续保留给人阅读。
    target_value: Optional[float] = None  # 仅人格连续维度可选；不是诊断分
    # Bounded engineering evidence accumulator, NOT a calibrated probability
    # or the model's current evidence-strength input. Old persisted values stay
    # readable and are not reinterpreted/rescaled by a schema migration.
    confidence: float = 0.0
    status: str = "candidate"             # candidate / working / stable / retired
    activation_epoch: int = 0
    first_observed_at: str
    last_confirmed_at: Optional[str] = None
    evidence: List[PersonHypothesisEvidence] = Field(default_factory=list)
    status_history: List[PersonHypothesisTransition] = Field(
        default_factory=list
    )
    atomic_claim: Optional[AtomicPersonClaim] = None

    def supporting_sessions(self) -> set[str]:
        return {
            item.session_id for item in self.evidence
            if item.relation == "support"
        }

    def counter_sessions(self) -> set[str]:
        return {
            item.session_id for item in self.evidence
            if item.relation == "counter"
        }

    def supporting_days(self) -> set[str]:
        """支持证据覆盖的自然日；稳定结论不能由同一天密集对话刷出来。"""
        return {
            item.at[:10] for item in self.evidence
            if item.relation == "support" and len(item.at) >= 10
        }

    def activation_evidence(self) -> List[PersonHypothesisEvidence]:
        """Evidence in the current activation attempt; all epochs stay auditable."""
        return [
            item for item in self.evidence
            if item.activation_epoch == self.activation_epoch
        ]

    def supporting_events(self) -> set[str]:
        """Strict event identities; an old session id is not a new event."""
        return {
            item.event_id for item in self.evidence
            if item.relation == "support" and item.event_id is not None
        }


class PersonModelState(BaseModel):
    """人物模型的演进元数据，用于证明经历了哪些独立会话，而非只展示最终分数。"""

    sessions_completed: int = 0
    completed_session_ids: List[str] = Field(default_factory=list)
    sessions_observed: int = 0
    reflected_session_ids: List[str] = Field(default_factory=list)
    first_interaction_at: Optional[str] = None
    last_reflected_at: Optional[str] = None


class RelationshipFactSource(BaseModel):
    """A verbatim relationship fact and its complete owner source turn."""

    quote: str
    turn_index: StrictInt = Field(ge=0)
    source_context: str
    source_context_sha256: str
    source_context_version: str


class RelationshipEvidence(BaseModel):
    """一条关系事实的来源；关系图不能靠模型臆测。"""

    at: str
    session_id: str
    summary: str
    turn_indices: List[int] = Field(default_factory=list)
    source: str = "user_conversation"
    # Old evidence summaries/indices alone do not preserve privacy context.
    fact_sources: List[RelationshipFactSource] = Field(default_factory=list)


class RelationshipContextEntry(BaseModel):
    """关于家人、朋友、照护人等真实关系的私有上下文。

    它与个人情景记忆、公共社会知识分开保存。默认只供本人对话内使用，
    `consent_to_share` 不会因为模型抽取而自动变成 True。
    """

    id: str
    name: str
    relation: Optional[str] = None
    facts: List[str] = Field(default_factory=list)
    referenceable: StrictBool = False
    consent_to_share: bool = False
    first_observed_at: str
    last_confirmed_at: Optional[str] = None
    evidence: List[RelationshipEvidence] = Field(default_factory=list)

    def can_reference(self) -> bool:
        from ..memory.referenceability import (
            must_not_reference, reference_context_is_valid,
        )
        if self.referenceable is not True or not self.facts:
            return False
        if must_not_reference(f"{self.name}{self.relation or ''}{' '.join(self.facts)}"):
            return False
        verified: set[str] = set()
        for evidence in self.evidence:
            if evidence.source != "user_conversation" or not evidence.session_id.strip():
                continue
            for source in evidence.fact_sources:
                if source.quote not in self.facts:
                    continue
                if (type(source.turn_index) is not int or source.turn_index < 0
                        or source.turn_index not in evidence.turn_indices
                        or not reference_context_is_valid(
                            source.source_context, source.source_context_sha256,
                            source.source_context_version, source.quote,
                        )):
                    return False
                verified.add(source.quote)
        return all(fact in verified for fact in self.facts)


class PersistentTaskRecord(BaseModel):
    """跨会话保留的未完成养老 Skill 任务；不等同于工具已执行。"""

    id: str
    skill_id: str
    intent: str
    goal: str
    status: str
    created_at: str
    updated_at: str
    slots: Dict[str, str] = Field(default_factory=dict)
    missing_slots: List[str] = Field(default_factory=list)
    required_capability: Optional[str] = None
    capability_available: bool = False
    consent_required: bool = False
    resume_status: Optional[str] = None
    history: List[str] = Field(default_factory=list)


class SessionTurnSnapshot(BaseModel):
    """未结束语义会话的一轮文本快照；不包含原始音视频。"""

    user: str
    agent: str


class CognitiveSourceTurn(BaseModel):
    """Separate, consent-scoped short-lived source; never a reflection input."""

    turn_index: StrictInt
    text: str
    at: str
    consent_at: str


class SessionCheckpointRecord(BaseModel):
    """异常退出恢复用的短期文字检查点；限量、过期并随画像加密/删除。"""

    session_id: str
    scene_id: str
    updated_at: str
    expires_at: Optional[str] = None
    turns: List[SessionTurnSnapshot] = Field(default_factory=list)
    cognitive_turns: List[CognitiveSourceTurn] = Field(default_factory=list)
    cognitive_session_at: Optional[str] = None
    # Legacy checkpoints retained only a filtered tail and cannot recover the
    # original source indices. New bounded complete checkpoints carry an exact
    # sparse slot map; omitted slots contain no saved sensitive text.
    original_turn_count: Optional[StrictInt] = None
    turn_indices: List[StrictInt] = Field(default_factory=list)
    memory_consolidated: bool = False
    reflection_completed: bool = False
    offline_finalize_pending: bool = False


class OfflineSessionRecord(BaseModel):
    """Sealed, bounded text-only work; excluded slots retain their original index."""

    session_id: str
    scene_id: str
    created_at: str
    expires_at: str
    turns: List[SessionTurnSnapshot] = Field(default_factory=list)
    cognitive_turns: List[CognitiveSourceTurn] = Field(default_factory=list)
    cognitive_session_at: Optional[str] = None
    excluded_turn_indices: List[int] = Field(default_factory=list)
    source_completeness: Literal["complete", "legacy_tail_unknown_indices"] = "complete"
    recovery_mode: Literal["text_only"] = "text_only"
    memory_completed: bool = False
    # Old jobs default to unreviewed; stage completion alone does not prove
    # semantic privacy screening was performed before reflection.
    memory_privacy_review_completed: bool = False
    reflection_completed: bool = False
    attempts: int = 0
    state: Literal["pending", "started", "exhausted", "needs_review"] = "pending"
    active_stage: Optional[Literal["memory", "reflection"]] = None
    last_attempt_at: Optional[str] = None
    last_error_code: Optional[str] = None


class OfflineSessionAuditRecord(BaseModel):
    """Bounded metadata after completion/expiry; never retains transcript text."""

    session_id: str
    scene_id: str
    created_at: str
    expires_at: str
    finished_at: str
    outcome: str
    source_completeness: str
    recovery_mode: Literal["text_only"] = "text_only"
    turn_count: int = 0
    excluded_turn_count: int = 0
    memory_completed: bool = False
    reflection_completed: bool = False
    attempts: int = 0
    gap: bool = False
    error_code: Optional[str] = None


class ReminderRule(BaseModel):
    """由护理员/本人明确设置的确定性提醒规则，不由对话模型猜测。"""

    id: str
    kind: str                         # medication / hydration / activity / appointment
    label: str
    at: str                           # 24 小时制 HH:MM
    recurrence: str = "daily"
    timezone: str = "Asia/Shanghai"
    active: bool = True
    created_by: str = "caregiver"
    require_ack: bool = True
    retry_after_minutes: int = 10
    max_attempts: int = 2
    escalation_after_minutes: int = 45
    contact_name: Optional[str] = None
    next_due_at: Optional[str] = None
    created_at: str
    updated_at: str


class ReminderOccurrenceRecord(BaseModel):
    """一次提醒的执行事实；未回应只记为 unconfirmed，绝不推断未完成。"""

    id: str
    reminder_id: str
    due_at: str
    # accepted 仅表示外部网关受理，不等于终端送达。
    status: str = "due"               # due/sent_local/sent/accepted/acknowledged/declined/completed/escalated
    sent_at: Optional[str] = None
    accepted_at: Optional[str] = None
    response_at: Optional[str] = None
    attempts: int = 0
    last_attempt_at: Optional[str] = None
    care_task_id: Optional[str] = None
    note: Optional[str] = None


class WellnessObservationRecord(BaseModel):
    """老人自愿表达或已授权来源提供的生活变化，不等同于医疗结论。"""

    id: str
    at: str
    metric: str                       # sleep_hours/appetite_score/pain_score/activity_score
    value: float
    unit: Optional[str] = None
    source: str = "user_report"
    confidence: float = 1.0
    baseline_delta: Optional[float] = None
    note: Optional[str] = None


class CognitiveHealthConsent(BaseModel):
    """认知健康敏感数据授权；与普通记忆、家属通知授权相互独立。"""

    brain_health_guidance_enabled: bool = True
    text_monitoring_enabled: bool = False
    text_consent_source: Optional[str] = None
    text_consented_at: Optional[str] = None
    text_revoked_at: Optional[str] = None
    # 首版不从声音推断认知变化；字段为未来经验证的适配器保留，默认关闭。
    audio_feature_monitoring_enabled: bool = False
    audio_consent_source: Optional[str] = None
    audio_consented_at: Optional[str] = None
    audio_revoked_at: Optional[str] = None

    @staticmethod
    def _valid_grant(
        enabled: bool,
        source: Optional[str],
        consented_at: Optional[str],
        revoked_at: Optional[str],
    ) -> bool:
        if not enabled or revoked_at or not (source or "").strip():
            return False
        raw = (consented_at or "").strip()
        if not raw:
            return False
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return False
        return parsed.utcoffset() is not None

    def has_text_monitoring_consent(self) -> bool:
        return self._valid_grant(
            self.text_monitoring_enabled,
            self.text_consent_source,
            self.text_consented_at,
            self.text_revoked_at,
        )

    def has_audio_feature_monitoring_consent(self) -> bool:
        return self._valid_grant(
            self.audio_feature_monitoring_enabled,
            self.audio_consent_source,
            self.audio_consented_at,
            self.audio_revoked_at,
        )


class CognitiveObservationRecord(BaseModel):
    """一条本人主动表达的认知/日常功能变化；不是量表结果或疾病标签。"""

    id: str
    at: str
    session_id: str
    domain: str                        # memory/orientation/language/executive/daily_function
    signal: str
    severity: str = "watch"           # watch / functional
    functional_impact: bool = False
    relative_to_personal_baseline: bool = False
    source: str = "explicit_user_report"
    confidence: float = 1.0
    confounders: List[str] = Field(default_factory=list)
    note: Optional[str] = None
    source_quote: Optional[str] = None
    source_turn_index: Optional[StrictInt] = None
    consent_at: Optional[str] = None
    event_id: Optional[str] = None
    counting: bool = True


class CognitiveTrendState(BaseModel):
    """透明的工程状态机；状态不是临床风险等级。"""

    status: str = "disabled"           # disabled/calibrating/watch/review_suggested
    episode_started_at: Optional[str] = None
    supporting_days: List[str] = Field(default_factory=list)
    domains: List[str] = Field(default_factory=list)
    active_task_id: Optional[str] = None
    updated_at: Optional[str] = None
    algorithm_version: str = "cognitive-change-v2-20261003"
    calibration_started_at: Optional[str] = None
    calibration_completed_at: Optional[str] = None
    # Session clocks are source capture clocks, never model-returned dates.
    session_times: Dict[str, str] = Field(default_factory=dict)
    cooldown_until: Optional[str] = None
    declines: int = 0
    last_offered_session: Optional[str] = None
    deleted_observation_ids: List[str] = Field(default_factory=list)
    timezone: str = "Asia/Shanghai"


class CareTaskTransition(BaseModel):
    at: str
    from_status: Optional[str] = None
    to_status: str
    actor: str
    note: Optional[str] = None


class CareTaskRecord(BaseModel):
    """需要真人处理的最小护理任务；不把聊天全文复制给第三方。"""

    id: str
    kind: str
    priority: str                     # low/medium/high/critical
    summary: str
    rationale: str
    # recorded_local 只代表写入本地开发者模拟器；accepted 只代表外部网关
    # 受理；notified 才代表适配器返回终端送达回执。三者不能合并成
    # “已通知真人”。
    status: str = "open"
    created_at: str
    updated_at: str
    assigned_contact: Optional[str] = None
    consent_basis: Optional[str] = None
    related_record_id: Optional[str] = None
    dedupe_key: Optional[str] = None
    attempts: int = 0
    next_retry_at: Optional[str] = None
    resolution: Optional[str] = None
    history: List[CareTaskTransition] = Field(default_factory=list)


class NotificationReceiptRecord(BaseModel):
    """外部或本地守护收件箱的真实投递回执。"""

    id: str
    task_id: Optional[str] = None
    occurrence_id: Optional[str] = None
    audience: str                      # elder / caregiver
    contact: str
    channel: str
    status: str                        # delivered_local/accepted/delivered/failed
    attempted_at: str
    delivered_at: Optional[str] = None
    provider_id: Optional[str] = None
    error: Optional[str] = None
    idempotency_key: str
    message: Optional[str] = None


class CarePlan(BaseModel):
    """结构化护理数据 —— 关于这个人的「事实」那一半(区别于情景记忆)。

    工具(吃药提醒 / 联系家人)的「数据」来自这里,不该写死在工具里:
    工具保持通用,「吃什么药、几点、联系谁」是这个人的知识。
    """
    conditions: List[str] = Field(default_factory=list)
    medications: List[Medication] = Field(default_factory=list)
    emergency_contacts: List[EmergencyContact] = Field(default_factory=list)


class PreferenceQuote(BaseModel):
    """A verbatim user turn, never an assistant interpretation."""

    model_config = ConfigDict(extra="forbid")
    session_id: str = Field(min_length=1, max_length=100)
    turn_index: StrictInt = Field(ge=0)
    text: str = Field(min_length=1, max_length=240)


class PreferenceUIConfirmation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: Literal["ui"] = "ui"
    at: str


class PreferenceTransition(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    at: str
    from_status: Optional[str] = Field(default=None, alias="from", serialization_alias="from")
    to: Literal["proposed", "confirmed", "declined", "withdrawn", "superseded"]
    source: Literal["rule", "model", "conversation", "ui"]
    quote: Optional[PreferenceQuote] = None


class PreferenceSemanticReview(BaseModel):
    """A model's bounded classification, not an independent truth or consent."""

    model_config = ConfigDict(extra="forbid")
    version: Literal["preference_semantics_v1"]
    kind: Literal["communication_style", "permission_boundary", "action_request",
                  "fact", "person_inference", "unclear"]
    owner_requested: StrictBool
    quote_supported: StrictBool
    scope_preserved: StrictBool
    no_permission_claim: StrictBool


class StatedPreference(BaseModel):
    """本人明确提出的交流说明；独立于人物推断，也不是分享授权。"""

    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^pref_[A-Za-z0-9_-]{1,59}$")
    do: str = Field(min_length=1, max_length=60)
    scope: str = Field(min_length=1, max_length=60)
    not_implies: List[str] = Field(default_factory=list, max_length=3)
    quotes: List[PreferenceQuote] = Field(min_length=1, max_length=2)
    confirmation: Optional[PreferenceQuote | PreferenceUIConfirmation] = None
    status: Literal["proposed", "confirmed", "declined", "withdrawn", "superseded"] = "proposed"
    supersedes: Optional[str] = Field(default=None, pattern=r"^pref_[A-Za-z0-9_-]{1,59}$")
    history: List[PreferenceTransition] = Field(default_factory=list)
    extraction_source: Literal["rule", "model"] = "model"
    # Missing on old profiles. Preserve history, but never infer that an old
    # confirmation also completed the current semantic admission contract.
    semantic_review: Optional[PreferenceSemanticReview] = None
    created_at: str
    updated_at: str

    @field_validator("do", "scope")
    @classmethod
    def instruction_is_not_a_trait_or_permission(cls, value: str) -> str:
        forbidden = ("disc", "big five", "big_five", "cbt", "依恋", "内向", "外向",
                     "神经质", "人格", "性格", "诊断", "因为", "导致", "所以", "缺乏安全感",
                     "分享", "转告", "通知", "联系家", "告诉家", "告诉女", "告诉儿",
                     "授权", "外发", "抑郁症", "焦虑症")
        if not value.strip() or any(word in value.lower() for word in forbidden):
            raise ValueError("preference_must_be_communication_instruction")
        return value

    @field_validator("not_implies")
    @classmethod
    def bounded_non_implications(cls, values: List[str]) -> List[str]:
        if any(not value.strip() or len(value) > 60 for value in values):
            raise ValueError("invalid_preference_non_implication")
        return values

    @field_validator("created_at", "updated_at")
    @classmethod
    def iso_time(cls, value: str) -> str:
        datetime.fromisoformat(value)
        return value

    @model_validator(mode="after")
    def confirmed_requires_confirmation(self):
        if self.status in {"confirmed", "withdrawn", "superseded"} and self.confirmation is None:
            raise ValueError("preference_confirmation_required")
        if self.supersedes == self.id:
            raise ValueError("preference_cannot_supersede_itself")
        return self


class PersonaProfile(BaseModel):
    """完整 persona 档案,从 yaml / json 加载。"""
    user_id: str
    profile_origin: str = "blank"          # blank / synthetic_demo / imported
    demographics: Optional[Demographics] = None
    background: Optional[Background] = None
    disc: DISC = Field(default_factory=DISC)
    big_five: BigFive = Field(default_factory=BigFive)
    attachment_style: Optional[str] = None
    cognitive_profile: CognitiveProfile = Field(default_factory=CognitiveProfile)
    case_conceptualization: CaseConceptualization = Field(default_factory=CaseConceptualization)
    behavioral_markers: BehavioralMarkers = Field(default_factory=BehavioralMarkers)
    emotional_baseline: EmotionalBaseline = Field(default_factory=EmotionalBaseline)
    episodic_memory: List[EpisodicMemoryEntry] = []
    aria_strategy_hints: AriaStrategyHints = Field(default_factory=AriaStrategyHints)
    care_plan: Optional[CarePlan] = None      # 结构化护理数据(用药 / 紧急联系人)
    alliance: AllianceRecord = Field(default_factory=AllianceRecord)      # 关系账本(跨会话)
    delivery_state: DeliveryModeState = Field(default_factory=DeliveryModeState)  # 交付滞回
    last_interaction_at: Optional[str] = None  # 最近一次互动时间(ISO);久未联系主动触发用
    risk_events: List[RiskEvent] = Field(default_factory=list)  # 历史/模型分流记录，不证明本人风险
    safety_screenings: List[SafetyScreeningRecord] = Field(default_factory=list)  # 主体未核实的兜底操作审计
    intervention_history: List[InterventionOutcomeRecord] = Field(default_factory=list)
    longitudinal_baseline: LongitudinalBaseline = Field(
        default_factory=LongitudinalBaseline
    )
    person_hypotheses: List[PersonHypothesis] = Field(default_factory=list)
    stated_preferences: List[StatedPreference] = Field(default_factory=list)
    evidence_review_ledger: List[PersonEvidenceReview] = Field(default_factory=list)
    person_model_state: PersonModelState = Field(default_factory=PersonModelState)
    relationship_context: List[RelationshipContextEntry] = Field(
        default_factory=list
    )
    active_tasks: List[PersistentTaskRecord] = Field(default_factory=list)
    reminder_rules: List[ReminderRule] = Field(default_factory=list)
    reminder_occurrences: List[ReminderOccurrenceRecord] = Field(
        default_factory=list
    )
    wellness_observations: List[WellnessObservationRecord] = Field(
        default_factory=list
    )
    cognitive_health_consent: CognitiveHealthConsent = Field(
        default_factory=CognitiveHealthConsent
    )
    cognitive_observations: List[CognitiveObservationRecord] = Field(
        default_factory=list
    )
    cognitive_trend: CognitiveTrendState = Field(
        default_factory=CognitiveTrendState
    )
    care_tasks: List[CareTaskRecord] = Field(default_factory=list)
    notification_receipts: List[NotificationReceiptRecord] = Field(
        default_factory=list
    )
    pending_session: Optional[SessionCheckpointRecord] = None
    offline_sessions: List[OfflineSessionRecord] = Field(default_factory=list)
    offline_session_audit: List[OfflineSessionAuditRecord] = Field(default_factory=list)

    def understanding_stage(self) -> str:
        """对外可解释的认识阶段，不把中性默认分数伪装成人格结论。"""
        statuses = {item.status for item in self.person_hypotheses}
        if "stable" in statuses:
            return "established"
        if "working" in statuses:
            return "emerging"
        if (
            self.person_model_state.sessions_completed
            or self.person_model_state.first_interaction_at
            or self.person_hypotheses
            or self.episodic_memory
        ):
            return "observing"
        return "cold_start"

    def referenceable_episodes(self) -> List[EpisodicMemoryEntry]:
        return [e for e in self.episodic_memory if e.can_reference()]

    def referenceable_relationships(self) -> List[RelationshipContextEntry]:
        return [e for e in self.relationship_context if e.can_reference()]

    def internal_episodes(self) -> List[EpisodicMemoryEntry]:
        """Aria 内化但不能挑明的。"""
        return [e for e in self.episodic_memory if not e.can_reference()]
