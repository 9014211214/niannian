"""档案透明度：老人可纠正摘要 + 开发者结构化技术视图。

设计立场:知情、可查阅、可删除。老人首先看到不污名化、可纠正的自然语言
摘要；研究者可展开结构化技术视图，不能把内部量表分数当成人的永久标签。
任何皮(场景)都可以把它接到自己的查阅入口;删除走 PersonaStore.delete。
"""
from __future__ import annotations

from ..context import EngineContext
from ..model.persona import PersonaProfile

_ATT_ZH = {
    "secure": "安全型", "anxious": "焦虑型",
    "avoidant": "回避型", "disorganized": "紊乱型",
}

_STAGE_ZH = {
    "cold_start": "我们刚认识，我还没有形成判断",
    "observing": "我还在听和观察，不急着下结论",
    "emerging": "我有了一些暂时理解，仍需要你纠正",
    "established": "有些沟通偏好经过多次交流仍较稳定，但仍可推翻",
}
_CARE_STATUS_ZH = {
    "open": "待处理",
    "needs_authorization": "等待授权，尚未发送",
    "recorded_local": "仅记录在本地模拟器，未联系真人",
    "accepted": "外部网关已受理，尚未确认终端送达",
    "notified": "外部适配器确认终端送达",
    "acknowledged": "已标记接单（以真实回执为准）",
    "in_progress": "处理中",
}


def render_profile_summary(
    persona: PersonaProfile,
    ctx: EngineContext | None = None,
) -> str:
    """给被陪伴者看的自然语言摘要；不显示量表名、诊断词或内部 tier。"""
    p = persona
    lines = [
        "━━ Aria 正在怎样认识你 ━━",
        "",
        _STAGE_ZH.get(
            p.understanding_stage(), "我仍在认识你，不把一次聊天当成定论"
        ),
        "下面都是可以纠正的暂时理解。哪里不对，请直接告诉我。",
    ]
    hypotheses = [
        item for item in p.person_hypotheses
        if item.status in {"working", "stable"}
    ][-6:]
    if hypotheses:
        lines.append("【我暂时理解的沟通偏好】")
        lines.extend(f"  - {item.claim}" for item in hypotheses)
    else:
        lines.append("【沟通偏好】还没有足够证据，我会继续听，不主动贴标签。")
    if p.evidence_review_ledger:
        lines.append(
            f"【尚待核实的理解与依据】本机另保留 {len(p.evidence_review_ledger)} 条有限文字依据，"
            "包括暂不下结论的原因；这些不是对你的定性，可随完整档案删除。"
        )

    lines.append(f"【本机档案中的经历与近况】目前共 {len(p.episodic_memory)} 条")
    for item in p.episodic_memory[-6:]:
        lines.append(f"  - {item.summary}")

    if p.reminder_rules:
        lines.append("【已经明确设置的提醒】")
        for item in p.reminder_rules[-6:]:
            state = "生效中" if item.active else "已停用"
            lines.append(f"  - {item.at} {item.label}（{state}）")

    cognitive_enabled = (
        p.cognitive_health_consent.has_text_monitoring_consent()
    )
    lines.append("【认知健康观察】这是变化提示，不是阿尔茨海默病预测或诊断")
    if cognitive_enabled:
        status = {
            "calibrating": "正在建立个人变化基线",
            "watch": "有跨日变化，继续观察",
            "review_suggested": "建议由专业人员进一步了解",
        }.get(p.cognitive_trend.status, "已启用")
        lines.append(
            f"  - 已经本人单独授权保存对话中主动表达的变化；{status}；"
            f"现有 {len(p.cognitive_observations)} 条最小记录"
        )
    else:
        lines.append("  - 未启用长期观察；本轮可以回答脑健康问题，但不会保存认知变化")

    contacts = (
        p.care_plan.emergency_contacts
        if p.care_plan is not None else []
    )
    if contacts:
        lines.append("【通知联系人授权】授权配置不代表已经发送过消息")
        for item in contacts:
            state = "已授权" if item.consent_to_notify else "未授权"
            scope = (
                "，仅限本地开发者模拟"
                if item.notification_consent_scope
                == "local_demo_care_notifications" else ""
            )
            lines.append(
                f"  - {item.name}（{item.relation or '关系未说明'}）："
                f"{state}{scope}"
            )

    open_tasks = [
        item for item in p.care_tasks
        if item.status not in {"resolved", "closed", "cancelled"}
    ]
    if open_tasks:
        lines.append("【尚待处理的照护记录】记录任务不等于真人已经收到")
        for item in open_tasks[-6:]:
            lines.append(
                f"  - {item.summary}（"
                f"{_CARE_STATUS_ZH.get(item.status, item.status)}）"
            )

    if p.pending_session is not None:
        lines.append(
            "【未结束的聊天】短期文字检查点已保存；它不代表人物理解已经更新。"
        )
    if p.offline_sessions:
        lines.append(
            f"【等待会后整理】{len(p.offline_sessions)} 个已结束会话尚未处理完成；"
            "不会混入新聊天。文本最多保留24小时，失败不会算作已经学会。"
        )
    if ctx is not None and ctx.formulation.hypotheses:
        lines.append("【本轮】我仍在比较几种解释，不把其中任何一种说成事实。")
    lines.extend([
        "",
        "你可以在页面中删除本机保存的画像、文本检查点、待处理会话、记忆、风险与照护记录。",
        "删除本机数据不能撤回已经提交给云服务或外部通道的数据。",
    ])
    return "\n".join(lines)


def _bar(v: float) -> str:
    n = max(0, min(10, round(v * 10)))
    return "█" * n + "░" * (10 - n) + f" {v:.2f}"


def render_profile(persona: PersonaProfile, ctx: EngineContext | None = None) -> str:
    """渲染主要结构化字段，供开发者审计；不是老人端人格结论。"""
    p = persona
    lines = [f"━━ 开发者技术审计视图(user={p.user_id})━━",
             "",
             (
                 "【人物理解进度】"
                 f"阶段 {p.understanding_stage()}；"
                 f"完成 {p.person_model_state.sessions_completed} 个语义会话；"
                 f"其中 {p.person_model_state.sessions_observed} 个已完成模型反思；"
                 f"档案来源 {p.profile_origin}"
             ),
             (
                 "【画像分数说明】空白档案中的 0.50 是中性占位，不是对你的判断；"
                 "新观察保存在下方可推翻假设中，不会偷偷改写成永久标签。"
             ),
             "【DISC 沟通/行为风格基线】",
             f"  支配 D    {_bar(p.disc.dominance)}",
             f"  影响 I    {_bar(p.disc.influence)}",
             f"  稳健 S    {_bar(p.disc.steadiness)}",
             f"  尽责 C    {_bar(p.disc.conscientiousness)}",
             "【Big Five 特质基线】",
             f"  开放性    {_bar(p.big_five.openness)}",
             f"  尽责性    {_bar(p.big_five.conscientiousness)}",
             f"  外向性    {_bar(p.big_five.extraversion)}",
             f"  宜人性    {_bar(p.big_five.agreeableness)}",
             f"  神经质    {_bar(p.big_five.neuroticism)}"]
    att = _ATT_ZH.get(p.attachment_style or "", None)
    lines.append(f"【依恋倾向】{att or '尚无判断'}")
    if p.cognitive_profile.distortion_tendencies:
        lines.append("【认知模式候选（非诊断）】")
        lines += [
            f"  - {item}"
            for item in p.cognitive_profile.distortion_tendencies
        ]
    if p.person_hypotheses:
        lines.append("【可推翻的人物假设】候选假设不会直接改变回应策略")
        for hypothesis in p.person_hypotheses[-20:]:
            scope = f"；范围:{hypothesis.scope}" if hypothesis.scope else ""
            support = len(hypothesis.supporting_sessions())
            days = len(hypothesis.supporting_days())
            counter = len(hypothesis.counter_sessions())
            lines.append(
                f"  - [{hypothesis.status}] {hypothesis.claim}"
                f"（置信 {hypothesis.confidence:.2f}；支持会话 {support}；"
                f"覆盖 {days} 天；反证会话 {counter}{scope}）"
            )
            for evidence in hypothesis.evidence[-2:]:
                relation = "支持" if evidence.relation == "support" else "反证"
                lines.append(f"      · {relation}:{evidence.summary}")
            if hypothesis.status_history:
                transition = hypothesis.status_history[-1]
                lines.append(
                    f"      · 最近迁移:{transition.from_status}→"
                    f"{transition.to_status}（{transition.reason}）"
                )
    if p.evidence_review_ledger:
        lines.append("【会后证据审查】包含不采纳/未知；关系与理由是系统评估，不是本人事实或授权")
        for review in p.evidence_review_ledger[-6:]:
            lines.append(f"  - [{review.disposition}] {review.applicability}/{review.relation}；"
                         f"事件={review.event_id}；{review.reason}")
            for quote in review.evidence:
                lines.append(f"      · 本人原话[{review.session_id}:{quote.turn_index}]:{quote.quote}")
    lines.append(f"【交付方式】{p.delivery_state.committed or '尚未定挡(默认自然)'}")
    lines.append(f"【关系】信任 {p.alliance.trust:.2f}(0-1)/ 历史破裂 {p.alliance.rupture_count} 次")

    if p.case_conceptualization.core_patterns:
        lines.append("【我观察到的模式】")
        lines += [f"  - {x}" for x in p.case_conceptualization.core_patterns]
    if p.case_conceptualization.triggers:
        lines.append("【在意的触发点】")
        lines += [f"  - {x}" for x in p.case_conceptualization.triggers]
    if p.case_conceptualization.automatic_thoughts_or_beliefs:
        lines.append("【自动想法或信念候选】")
        lines += [
            f"  - {x}"
            for x in p.case_conceptualization.automatic_thoughts_or_beliefs
        ]
    if p.case_conceptualization.maintenance_factors:
        lines.append("【可能的维持因素】")
        lines += [
            f"  - {x}"
            for x in p.case_conceptualization.maintenance_factors
        ]
    if p.case_conceptualization.protective_factors:
        lines.append("【保护因素】")
        lines += [
            f"  - {x}"
            for x in p.case_conceptualization.protective_factors
        ]
    if ctx is not None and ctx.formulation.hypotheses:
        lines.append(f"【本次会话的工作假设】(置信 {ctx.formulation.confidence:.2f})")
        lines += [f"  - {x}" for x in ctx.formulation.hypotheses]

    lines.append(f"【记得你的事】共 {len(p.episodic_memory)} 条")
    for e in p.episodic_memory:
        flag = "可提及" if e.referenceable else "只内化"
        lines.append(f"  - [{flag}] {e.summary}")
    if p.relationship_context:
        lines.append("【你告诉我的关系】默认不代表允许分享给第三方")
        for relationship in p.relationship_context:
            flag = "可在对话中提及" if relationship.referenceable else "只内化"
            sharing = "允许分享" if relationship.consent_to_share else "不可外发"
            lines.append(
                f"  - [{flag}/{sharing}] {relationship.name}"
                f"（{relationship.relation or '关系未说明'}）："
                + "；".join(relationship.facts)
            )
    if p.care_plan is not None:
        if p.care_plan.medications:
            lines.append("【护理计划中的药物事实】不表示 Aria 可以调整药物")
            for medication in p.care_plan.medications:
                lines.append(
                    f"  - {medication.name}；{medication.dosage or '剂量未填'}；"
                    f"{medication.at or medication.schedule or '时间未填'}"
                )
        if p.care_plan.emergency_contacts:
            lines.append("【通知联系人授权】授权配置不表示消息已经发送")
            for contact in p.care_plan.emergency_contacts:
                state = "已授权" if contact.consent_to_notify else "未授权"
                lines.append(
                    f"  - {contact.contact_id or '尚无稳定ID'} "
                    f"{contact.name}/{contact.relation or '关系未说明'}：{state}；"
                    f"范围={contact.notification_consent_scope or '未记录'}；"
                    f"来源={contact.notification_consent_source or '未记录'}；"
                    f"授权时间={contact.notification_consented_at or '未记录'}；"
                    f"撤回时间={contact.notification_consent_revoked_at or '未记录'}"
                )
    if p.active_tasks:
        lines.append("【尚未结束的任务】任务存在不代表外部动作已经执行")
        for task in p.active_tasks:
            lines.append(
                f"  - [{task.status}] {task.goal}"
                + (
                    f"；尚缺:{'、'.join(task.missing_slots)}"
                    if task.missing_slots else ""
                )
                + (
                    f"；需要能力:{task.required_capability}"
                    if task.required_capability else ""
                )
            )
    if p.reminder_rules:
        lines.append("【确定性提醒】由本人或护理员明确设置")
        for rule in p.reminder_rules:
            lines.append(
                f"  - [{'生效' if rule.active else '停用'}] "
                f"{rule.at} {rule.label}；需要确认:{rule.require_ack}"
            )
    if p.reminder_occurrences:
        lines.append("【提醒回执】未回应只表示尚未确认，不代表没有完成")
        for occurrence in p.reminder_occurrences[-20:]:
            lines.append(
                f"  - {occurrence.due_at} [{occurrence.status}] "
                f"尝试 {occurrence.attempts} 次"
            )
    if p.wellness_observations:
        lines.append("【你主动表达的生活变化】变化提示不是疾病诊断")
        for item in p.wellness_observations[-20:]:
            lines.append(
                f"  - {item.at} {item.metric}={item.value:g}"
                f"{item.unit or ''}（来源:{item.source}）"
            )
    cognitive_consent = p.cognitive_health_consent
    lines.append("【认知健康敏感数据】与普通记忆、人格和联系人授权分开")
    lines.append(
        "  - 文本变化观察="
        f"{cognitive_consent.has_text_monitoring_consent()}；"
        f"授权时间={cognitive_consent.text_consented_at or '未记录'}；"
        f"撤回时间={cognitive_consent.text_revoked_at or '未记录'}"
    )
    lines.append(
        "  - 声音特征观察="
        f"{cognitive_consent.has_audio_feature_monitoring_consent()}；"
        "首版不从声音、情绪或人格推断认知变化"
    )
    lines.append(
        f"  - 工程状态={p.cognitive_trend.status}；"
        f"支持日期={len(p.cognitive_trend.supporting_days)}；"
        f"领域={','.join(p.cognitive_trend.domains) or '无'}；疾病概率=不输出"
    )
    if p.cognitive_observations:
        lines.append("【本人主动表达的认知/熟悉功能变化】本说明不含原话；原话仅在本人独立面板查看和删除")
        for item in p.cognitive_observations[-20:]:
            confounders = ",".join(item.confounders) or "无已知混杂"
            lines.append(
                f"  - {item.at} {item.domain}/{item.signal}；"
                f"影响熟悉功能={item.functional_impact}；混杂={confounders}"
            )
    if p.care_tasks:
        lines.append("【真人守护任务】只包含结构化最小信息，不复制聊天全文")
        for task in p.care_tasks[-20:]:
            lines.append(
                f"  - [{task.status}/{task.priority}] {task.summary}"
            )
    if p.notification_receipts:
        lines.append("【守护投递回执】本地收件箱不等于已联系真实护理员")
        for receipt in p.notification_receipts[-20:]:
            lines.append(
                f"  - {receipt.attempted_at} [{receipt.status}] "
                f"{receipt.audience}/{receipt.channel}"
            )
    if p.pending_session is not None:
        lines.append(
            "【待恢复会话】"
            f"{len(p.pending_session.turns)} 轮文字检查点；"
            "不含原始音视频，正常结束后会清除"
        )
    if p.offline_sessions:
        lines.append(
            f"【会后待处理队列】{len(p.offline_sessions)} 个已封存会话；"
            "按时间顺序处理，最多3次尝试，固定24小时保留；"
            "文本恢复不冒充声音或视觉观测。"
        )
        for job in p.offline_sessions:
            lines.append(f"  - {job.state} / 已尝试 {job.attempts} 次 / 到期 {job.expires_at}")
    if p.risk_events:
        lines.append(f"【安全记录】共 {len(p.risk_events)} 条(我担心过你的时刻,同样对你不保留)")
        for ev in p.risk_events:
            note = f":{ev.note}" if ev.note else ""
            lines.append(f"  - {ev.at} [tier{ev.tier} {ev.kind}]{note}")
    if p.longitudinal_baseline.signals:
        lines.append("【个人时间基线】只保存聚合统计，不保存原始音视频")
        for signal, stat in sorted(p.longitudinal_baseline.signals.items()):
            lines.append(
                f"  - {signal}: 历史 {stat.count} 次，均值 {stat.mean:.3f}"
            )
    if p.intervention_history:
        lines.append(f"【回应结果账本】共 {len(p.intervention_history)} 条")
        for result in p.intervention_history[-10:]:
            lines.append(
                f"  - {result.at} {result.intervention_label} → "
                f"{result.outcome}/{result.signal}"
            )
    if p.aria_strategy_hints.tactics:
        lines.append("【从重复结果中学到的方式】")
        lines += [f"  - {item}" for item in p.aria_strategy_hints.tactics]
    lines.append("")
    lines.append(
        "以上是供开发者审计的主要结构化字段，不是诊断，也不等于对人的永久结论。"
    )
    lines.append("Web 体验请使用“删除本机数据”按钮执行整体本地删除。")
    return "\n".join(lines)
