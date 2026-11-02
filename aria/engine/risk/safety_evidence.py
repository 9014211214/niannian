"""Candidate source/shape gates, never lexical diagnosis or tool permission."""
import hashlib

CONCERNS = {'none', 'self_harm', 'medical_emergency', 'fall', 'unsafe_medication'}
SUBJECTS = ('not_applicable', 'owner', 'other', 'quotation_or_hypothetical', 'uncertain')
TIMES = ('not_applicable', 'current', 'recent_resolved', 'historical', 'uncertain')
URGENCIES = ('not_applicable', 'immediate', 'not_immediate', 'uncertain')
UNCERTAIN_REPLY = (
    '我还不能确定是否有正在发生的危险。现在有人受伤、已误服药物，或可能马上伤害自己吗？'
    '若有眼前危险，请立即联系现场可信的人和当地紧急服务；不用等我继续回复。'
)
THIRD_PERSON_REPLY = (
    '这可能涉及当事人眼前的安全。请立即联系当地紧急服务，说明当事人的情况和位置；'
    '能联系到现场可信的人时请其协助，先保证你自己安全。我没有通知任何人。'
)
EVIDENCE_PROMPT = (
    '\n\n## 安全证据候选：safety_evidence_v1\n'
    'safety_evidence必填，quote为当前用户消息中的逐字锚点（最多240字符）；'
    '可结合必要近文理解“还没好”等指代，不引用助手文字或拼造引文。'
    'subject=owner本人/other第三人（包括远程转述）/quotation_or_hypothetical引述假设/uncertain；'
    'time=current当前危险/recent_resolved刚缓解/historical过去/uncertain；'
    'urgency=immediate需立即真人帮助/not_immediate非立即/uncertain。'
    '有无危险由语义判断，区分正在发生的事实、讨论与否认，不用相邻领域联想补出危险。'
    '无当前风险时safety_concern=none，quote为空，subject/time/urgency都为not_applicable；'
    '非none要给有效原话并如实填未知，不能为满足结构假造紧迫性。'
    '第三人当前危险也应准确识别，subject必须是other，不归给用户本人。'
    '先理解用户当下要解决的事情；普通任务与已解决过去事件不因相关词进入安全接管。'
    '风险分类必须对应用户实际陈述的危险事实或仍在持续的状态：'
    '分别辨明是谁、是否真实发生、是否仍在发生及是否需要即时帮助；'
    '不要把用户想完成的普通事情、修辞、担心一种可能性或明确否定的行为改写成已发生的危机。'
    '否定或已解决只作用于它明确指向的事实，不能因此漏掉同句话中另一个仍在发生的危险；'
    '信息不足时如实标uncertain，不用immediate填补未知。'
    'response仍须是完整、直接回应当前诉求的可交付正文：有实际危险时给针对当事人的安全帮助，'
    '无危险时完成原任务，不附加无关危机盘问；遵守用户明确的语言、句数和表达形式。'
    '只使用本轮实际提供的事实，不补造人的身份、经历、偏好、记录、邀请或权限。'
    'preference_control只用于页面已实际交付且本轮有效的待确认邀请；没有该邀请时省略整个字段，'
    '不要为了填满可选结构生成空白确认，也不要声称已保存、启用、拒绝或通知。'
    '非none由受控流程处理：证据不明用条件式确认，不把标签当诊断，'
    '不执行原答或工具，不产生护理授权；第三人危险不成为本人人物或护理证据。'
)

EVIDENCE_PROMPT_V2 = EVIDENCE_PROMPT.replace(
    '不执行原答或工具，不产生护理授权；第三人危险不成为本人人物或护理证据。',
    '可交付经过既有守卫的完整 response 正文，但不执行工具、不产生护理授权；第三人危险不成为本人人物或护理证据。',
)

def evidence_schema():
    return {'type': 'object', 'properties': {
        'quote': {'type': 'string', 'maxLength': 240},
        'subject': {'type': 'string', 'enum': list(SUBJECTS)},
        'time': {'type': 'string', 'enum': list(TIMES)},
        'urgency': {'type': 'string', 'enum': list(URGENCIES)},
    }, 'required': ['quote', 'subject', 'time', 'urgency'], 'additionalProperties': False}

def assess_evidence(concern, row, utterance):
    result = {'version': 'safety_evidence_v1', 'disposition': 'uncertain',
              'reason': 'invalid_evidence', 'verbatim_source': False,
              'semantic_verified': False, 'grants_permission': False}
    if (type(concern) is not str or concern not in CONCERNS or type(row) is not dict
            or set(row) != {'quote', 'subject', 'time', 'urgency'}
            or type(row['quote']) is not str or len(row['quote']) > 240
            or row['subject'] not in SUBJECTS or row['time'] not in TIMES
            or row['urgency'] not in URGENCIES):
        return result
    if concern == 'none':
        if row['quote'] == '' and all(row[key] == 'not_applicable' for key in ('subject', 'time', 'urgency')):
            result.update(disposition='clear', reason='model_none_structurally_consistent')
        else:
            result['reason'] = 'none_evidence_contradiction'
        return result
    quote = row['quote']
    if not quote.strip() or type(utterance) is not str or quote not in utterance:
        result['reason'] = 'current_quote_not_verbatim'
        return result
    result.update(verbatim_source=True, quote_sha256=hashlib.sha256(quote.encode('utf-8')).hexdigest())
    if 'not_applicable' in (row['subject'], row['time'], row['urgency']):
        result['reason'] = 'concern_evidence_contradiction'
    elif row['time'] == 'current' and row['urgency'] == 'immediate' and row['subject'] in {'owner', 'other'}:
        result.update(disposition='urgent_' + row['subject'], reason='model_current_immediate_with_source')
    else:
        result['reason'] = 'current_subject_or_urgency_not_established'
    return result
