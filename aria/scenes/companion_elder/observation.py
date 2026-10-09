"""养老场景的主动观察策略。

主动察言观色不是不断追问，也不是必须开摄像头。它是在竞争假设之间选择一个
最低打扰、最能减少关键不确定性的动作：问一句、继续观察，或明确停下来。
"""
from __future__ import annotations

import re

from ...engine.alliance import rejection_surface
from ...engine.context import EngineContext
from ...interfaces.reasoning import ActiveObservationPlan, WorkingHypothesis

_REJECTION = ("别问", "不想说", "烦不烦", "别管", "不用你管")
_EXPLICIT_PAUSE = re.compile(r"不想(?:展开|继续|详细|多|再)(?:说|聊)")
_DIRECT_REASON = ("因为", "其实就是", "就是觉得", "主要是", "说白了")
_QUOTED = re.compile(r'“[^”]*”|‘[^’]*’|「[^」]*」|"[^"\n]*"')
_SITUATIONAL_SOUND = re.compile(
    r"(?:窗外|外面|隔壁|楼上|周围|背景|施工|装修|电视).{0,12}(?:吵|噪声|噪音|杂音|太响)"
    r"|(?:麦克风|收音|信号|网络).{0,10}(?:不好|不稳|断了|杂音|出问题|有问题)"
)
_SELF_AFFECT_CORRECTION = re.compile(
    r"(?:不是|并不是|并非)(?:我|我在|我有)(?:生气|难过|紧张|不高兴|情绪不好)"
    r"|我(?:不是|并不是|并非)(?:在|有)?(?:生气|难过|紧张|不高兴|情绪不好)"
    r"|我(?:并没|并没有|没|没有)(?:在)?(?:生气|难过|紧张|不高兴)"
)
_NEW_DISTRESS_DISCLOSURE = re.compile(
    r"(?:但|不过|可是|现在|其实|而是|是我|只是我).{0,10}(?:难过|害怕|担心|想哭|难受|心里.{0,6}(?:事|压|难))"
)
_CURRENT_SOUND_RECOVERY = re.compile(
    r"(?:现在|这会儿|已经).{0,6}(?:安静了|不吵了|听清了|声音正常|收音正常|网络好了)"
)


def _direct_situational_correction(utterance: str) -> bool:
    """Finite user correction of a sound interpretation, not an all-clear.

    A named current environment/device explanation AND a first-person affect
    correction are required. Quoted, reported, hypothetical and old events are
    not current self evidence. This does not delete sensor data or hypotheses.
    """
    text = _QUOTED.sub("", utterance)
    if any(marker in text for marker in (
        "他说", "她说", "他们说", "别人说", "新闻里", "电视里", "故事里", "如果", "假如",
        "以前", "昨天", "上次", "去年", "那时候", "我不是说", "不能说", "不代表", "未必",
        "不是我没有", "并非我没有", "不是我没", "并非我没",
    )):
        return False
    return bool(_SITUATIONAL_SOUND.search(text) and _SELF_AFFECT_CORRECTION.search(text))


def _current_sound_explanation(ctx: EngineContext, utterance: str) -> bool:
    if _NEW_DISTRESS_DISCLOSURE.search(_QUOTED.sub("", utterance)):
        return False
    if _direct_situational_correction(utterance):
        return True
    # A next-turn explicit confirmation of recovery may finish the same
    # explanation. Do not create a persistent "fine" flag or suppress a later
    # unrelated request because of an old correction somewhere in history.
    return bool(
        ctx.turns and _direct_situational_correction(ctx.turns[-1].user)
        and _CURRENT_SOUND_RECOVERY.search(_QUOTED.sub("", utterance))
    )


def _has_current_rejection(utterance: str) -> bool:
    """A narrow inverse-negation exception, not sentiment interpretation.

    Remove only an explicitly rejected interpretation of silence. A separate
    real stop remains a stop, including a stop following a contrast. Do not
    infer permission from unrelated negation or from a personality hypothesis.
    """
    compact = rejection_surface(utterance)
    return any(word in compact for word in _REJECTION) or bool(_EXPLICIT_PAUSE.search(compact))

_PAIR_QUESTIONS = {
    frozenset(("reduced_life_engagement", "physical_discomfort")): (
        "最近不太想出门，是身体不太方便，还是去了也觉得没什么意思？"
    ),
    frozenset(("reduced_life_engagement", "loss_of_role")): (
        "是最近提不起兴致，还是觉得去了也帮不上什么忙？"
    ),
    frozenset(("social_disconnection", "loss_of_role")): (
        "你更像是不想见那些人，还是觉得自己去了也没什么可做？"
    ),
    frozenset(("concealed_distress", "physical_discomfort")): (
        "我不急着问心情——是身体有点不舒服，还是最近碰上什么事了？"
    ),
    frozenset(("autonomy_threat", "reduced_life_engagement")): (
        "你更不想去，是因为这事本身没意思，还是大家已经替你安排好了？"
    ),
    frozenset(("grief_or_transition", "social_disconnection")): (
        "更像是想念一个具体的人，还是最近能说上话的人都少了？"
    ),
}

_SINGLE_QUESTIONS = {
    "concealed_distress": "我不急着追问，是身体有点不舒服，还是最近有件事压在心里？",
    "reduced_life_engagement": "最近不太想参与，是身体累，还是觉得那些事没什么意思了？",
    "loss_of_role": "你说帮不上忙，是觉得做不了，还是觉得他们已经不需要你了？",
    "social_disconnection": "你是不太想参加活动，还是有哪个人让你不太想碰面？",
    "physical_discomfort": "这个不舒服是刚刚才有，还是已经持续一阵了？",
    "autonomy_threat": "你更反感他们反复催，还是他们已经替你作了决定？",
    "grief_or_transition": "你更想说说发生的事，还是先让我陪你待一会儿？",
}


class ElderActiveObservationPolicy:
    """在不丢面子、不连续盘问的前提下，为下一拍选择求证方式。"""

    def plan(
        self,
        ctx: EngineContext,
        utterance: str,
        hypotheses: list[WorkingHypothesis],
        *,
        selected_intervention_id: str = "",
        person_supported_route: bool = False,
        verified_cooldown_routes: tuple[str, ...] = (),
        current_user_reopened: bool = False,
        current_physical_evidence_qualified: bool = False,
    ) -> ActiveObservationPlan:
        current_turn = len(ctx.turns)
        if ctx.working_memory.get("risk_flag"):
            return ActiveObservationPlan(
                mode="wait",
                reason="关系或安全信号优先；这一拍停止收集信息，先维持安全与自主。",
            )
        if (
            ctx.working_memory.get("_online_decision_version") != "online_turn_decision_v3"
            and
            not current_user_reopened
            and (
                ctx.alliance.ruptured
                or _has_current_rejection(utterance)
            )
        ):
            return ActiveObservationPlan(
                mode="wait",
                reason="关系或安全信号优先；这一拍停止收集信息，先维持安全与自主。",
            )
        if verified_cooldown_routes:
            return ActiveObservationPlan(
                mode="wait",
                reason=(
                    "相关沟通方向已在至少两个独立会话得到明确负面结果；"
                    "本人本轮没有重新提出该需要，这一拍停止重复推动。"
                ),
                hypothesis_ids=[item.id for item in hypotheses[:2]],
            )
        if (
            not current_physical_evidence_qualified
            and not current_user_reopened
            and _current_sound_explanation(ctx, utterance)
        ):
            return ActiveObservationPlan(
                mode="observe",
                reason=(
                    "本人已把这次声音解释为环境或设备因素，并纠正情绪解读；"
                    "先接住解释或紧接着的恢复确认，不重新强制身体/心事二选一。"
                    "这不证明情绪正常，也不取消新的身体事实或明确求助。"
                ),
                hypothesis_ids=[item.id for item in hypotheses[:2]],
            )
        if not hypotheses:
            return ActiveObservationPlan(
                mode="observe",
                reason="当前证据不足，不为了展示能力强行提问。",
            )
        previous_execution = (
            ctx.reasoning.observation_history[-1]
            if ctx.reasoning.observation_history else None
        )
        if (
            previous_execution is not None
            and previous_execution.turn_index == current_turn - 1
            and previous_execution.executed_mode == "ask"
        ):
            return ActiveObservationPlan(
                mode="observe",
                reason="上一拍已经主动求证，本轮先看用户如何回应，避免连续盘问。",
                hypothesis_ids=[item.id for item in hypotheses[:2]],
            )

        # 人物模型已经把一个有明确做法的沟通方向排到首位时，不再让通用的
        # “二选一求证题”覆盖它。这里仍是 observe，而非绝对禁止日常邀请：
        # 最终回应可以自然留一个可拒绝入口，但不必先完成一张诊断问卷。
        # 身体澄清例外，它依赖当前身体证据，仍需保留具体求证问题。
        if (
            person_supported_route
            and selected_intervention_id
            and selected_intervention_id != "physical_clarification"
        ):
            return ActiveObservationPlan(
                mode="observe",
                reason=(
                    "跨会话人物证据已支持当前沟通方向；先执行该方向并观察结果，"
                    "不强制追加通用诊断式求证。"
                ),
                hypothesis_ids=[item.id for item in hypotheses[:2]],
            )
        if any(marker in utterance for marker in _DIRECT_REASON):
            return ActiveObservationPlan(
                mode="observe",
                reason="用户已经主动给出原因，先接住并观察，不把披露变成问卷。",
                hypothesis_ids=[item.id for item in hypotheses[:2]],
            )

        top = hypotheses[0]
        second_confidence = (
            hypotheses[1].confidence if len(hypotheses) > 1 else 0.0
        )
        if (
            top.confidence < 0.32
            and second_confidence < 0.28
        ) or (
            top.counterevidence_ids
            and top.confidence < 0.50
        ):
            return ActiveObservationPlan(
                mode="observe",
                reason=(
                    "当前只有低置信线索或同时存在反证；先回应事实，"
                    "不把模型的不确定性转嫁成对老人的追问。"
                ),
                hypothesis_ids=[item.id for item in hypotheses[:2]],
            )
        if top.confidence >= 0.82 and not top.counterevidence_ids:
            return ActiveObservationPlan(
                mode="observe",
                reason="当前方向证据相对集中，先回应并观察结果，不额外追问。",
                hypothesis_ids=[top.id],
            )

        contenders = [
            item for item in hypotheses[1:4]
            if abs(top.confidence - item.confidence) <= 0.28
        ]
        second = contenders[0] if contenders else None
        question = ""
        ids = [top.id]
        if second is not None:
            ids.append(second.id)
            question = _PAIR_QUESTIONS.get(frozenset((top.id, second.id)), "")
        if not question:
            question = _SINGLE_QUESTIONS.get(top.id, "")
        if not question:
            return ActiveObservationPlan(
                mode="observe",
                reason="没有足够低打扰的区分问题，保留假设并继续观察。",
                hypothesis_ids=ids,
            )

        gap = abs(top.confidence - second.confidence) if second else 0.35
        uncertainty = 1.0 - abs(top.confidence - 0.5) * 2.0
        information_gain = max(
            0.0, min(1.0, uncertainty * 0.65 + (1.0 - gap) * 0.35)
        )
        return ActiveObservationPlan(
            mode="ask",
            reason="用一个二选一事实问题区分竞争解释，不要求用户承认某种情绪。",
            question=question,
            hypothesis_ids=ids,
            information_gain=information_gain,
            intrusiveness="low",
        )
