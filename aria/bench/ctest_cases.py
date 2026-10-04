"""Authored fictional C-test scripts; no real people, dates, or model replies.

The virtual calendar exists only inside this component experiment. These
scripts do not add backdated sessions to a user or rewrite production logs.
Group labels are evaluation metadata, never sent to the extractor.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import json


VERSION = "ctest_fictional_v1_draft"
DAY_OFFSETS = (0, 2, 4, 7, 10, 12, 18, 24)
VIRTUAL_EPOCH = datetime(2000, 1, 1, 9, tzinfo=timezone(timedelta(hours=8)))


@dataclass(frozen=True)
class CTestSession:
    number: int
    virtual_day: int
    utterances: tuple[str, ...]
    synthetic_assistant: tuple[str, ...]

    @property
    def virtual_at(self):
        return VIRTUAL_EPOCH + timedelta(days=self.virtual_day)


@dataclass(frozen=True)
class CTestCase:
    id: str
    group: str
    name: str
    sessions: tuple[CTestSession, ...]


_PEOPLE = (
    ("c01", "change", "林师傅", "修收音机", "旧收音机", "我以前在电器厂上班。"),
    ("c02", "change", "何阿姨", "养兰花", "那盆兰花", "我家的阳台种了好几盆花。"),
    ("c03", "change", "高师傅", "下象棋", "象棋盘", "以前下班了常跟同事下一盘棋。"),
    ("c04", "change", "朱阿姨", "缝布袋", "那块蓝布", "我做衣服做了三十多年。"),
    ("c05", "change", "沈师傅", "写毛笔字", "那支毛笔", "我喜欢慢慢写几个字，图个清静。"),
    ("c06", "change", "陶阿姨", "做面点", "那只蒸笼", "我以前最拿手的是蒸花卷。"),
    ("n01", "control", "徐师傅", "拍花草", "旧相机", "我出门常带相机拍些花草。"),
    ("n02", "control", "方阿姨", "钩围巾", "那团毛线", "天冷了我就喜欢做点针线活。"),
    ("n03", "control", "马师傅", "听评书", "小音箱", "评书我从年轻时候就爱听。"),
    ("n04", "control", "邵阿姨", "种小葱", "长花盆", "阳台那一角我种了几排小葱。"),
    ("n05", "control", "姜师傅", "做木工", "小木凳", "我喜欢修修家里的小物件。"),
    ("n06", "control", "叶阿姨", "练合唱", "歌本", "我每周去社区和大家唱唱歌。"),
)

# Later change sessions contain two different-day functional impacts. With six
# baseline visits and only eight visits total, this set does not independently
# measure the three-post-baseline-day growth branch of the production rules.
_IMPACTS = (
    ("我又忘了交电费，账单过期了，这事耽误了家里的安排。", "我这次又忘了交水费，过了缴费日，耽误了用水手续。"),
    ("我买菜时算不清账，最后只能少买几样，做饭的安排受影响了。", "这两天买菜还是算不清账，我没法把这一周的钱理明白。"),
    ("我忘了交水费，拖到过期了，这已经影响家里办事了。", "我又忘了交电费，过了时间，这次还是耽误了事情。"),
    ("我做饭总忘了先洗菜还是先切菜，没法照平时的步骤做好。", "这回我做饭又忘了已经放过什么材料，没法做成平常那道菜。"),
    ("我又忘了买菜，做饭时没材料，原定的饭耽误了。", "我忘了交电费，等要办手续才发现过期，又耽误事情了。"),
    ("我看账单时算不清总数，没法把家里的账整理好。", "今天买菜仍算不清账，付钱这一步就耽误了很久。"),
)
_CONTROL_DETAILS = (
    ("我一时想不起钥匙放哪儿了，后来在外衣口袋找到了。", "钥匙现在放门边，今天出门很顺当。"),
    ("我昨晚没睡好，早上想不起眼镜放哪儿了，睡醒就找到了。", "这两天睡得足，早上的事都照常做了。"),
    ("我故意又讲了一遍年轻时坐火车的故事，跟你说过吧。", "那个故事我有意再说一遍，就是爱说那一段。"),
    ("我说那个东西叫啥来着，那是方言一时没换成普通话。", "今天说快了有个口误，那个菜名我还是知道的。"),
    ("护理提醒显示未确认，我当时在做木工，手上不方便点。", "今天那条提醒我主动取消了，因为原来的事改期了。"),
    ("我没忘交电费，刚才只是把日期说错了，账单已经办好。", "我没有搞混日期，刚刚没听清你说的是哪一天。"),
)
_DAYS = (
    ("今天太阳挺好，我把窗户开了一会儿。", "中午做了面条，留了一点明天吃。"),
    ("早上有点风，我在楼下走了一小圈。", "买了几个苹果，洗好了放在桌上。"),
    ("今天天气阴着，我就没走远。", "早饭吃了粥和一个鸡蛋。"),
    ("刚晒完被子，摸起来暖暖的。", "午饭的青菜炒得正好。"),
    ("楼下新开的小店我路过看了看。", "晚上打算做个番茄汤。"),
    ("今天收拾柜子，翻出几张老照片。", "我煮了点杂粮饭，味道还行。"),
    ("早上坐在窗边，看见两只小鸟。", "中午还有半盘菜，晚上热一热。"),
    ("这几天转暖了，我换了件薄外套。", "刚烧好一壶水，坐下歇一歇。"),
)


def load_ctest_cases() -> tuple[CTestCase, ...]:
    """Materialize authored scripts; every visit has exactly ten user turns."""
    cases = []
    for index, (identifier, group, name, activity, object_name, background) in enumerate(_PEOPLE):
        sessions = []
        for number, day in enumerate(DAY_OFFSETS, 1):
            weather, meal = _DAYS[number - 1]
            detail = f"我把{object_name}拿出来看看，慢慢弄，不着急。"
            second = "我想先照自己的节奏过今天。"
            if group == "change":
                if number == 3:
                    detail = "我一时想不起剪刀放哪儿了，后来在抽屉找到了。"
                elif number == 5:
                    detail = "这几天我想不起放东西的位置，找了很久才找到。"
                elif number == 6:
                    detail = "昨天又想不起茶杯放哪儿了，我在屋里转了好几圈。"
                elif number in (7, 8):
                    detail = _IMPACTS[index][number - 7]
                    second = ("我又想不起眼镜放哪儿了，找东西花的时间多了。" if number == 7 else
                              "那个东西叫什么，我知道它干什么，话到嘴边却说不出名字。")
            elif number in (3, 7, 8):
                detail = _CONTROL_DETAILS[index - 6][0 if number in (3, 7) else 1]
            words = (
                "你好，坐下聊几句。", weather,
                f"我叫{name}，{background}" if number == 1 else f"今天还惦记着{activity}那点事。",
                meal, detail, f"说起{activity}，我还是愿意慢慢弄明白。", second,
                "我不赶时间，聊完还要把桌面理一理。", f"{object_name}就先放在桌边。",
                "今天说到这儿吧，我去歇会儿。",
            )
            assistants = tuple(("嗯，我听着。", "你按自己的节奏说。", "好，我们接着聊。")[i % 3]
                               for i in range(len(words)))
            sessions.append(CTestSession(number, day, words, assistants))
        cases.append(CTestCase(identifier, group, name, tuple(sessions)))
    return tuple(cases)


def ctest_manifest(cases: tuple[CTestCase, ...]) -> dict:
    """Evaluator-only inventory; never pass this object to a model client."""
    from dataclasses import asdict
    if (len(cases) != 12 or len({case.id for case in cases}) != 12
            or Counter(case.group for case in cases) != {"change": 6, "control": 6}):
        raise ValueError("C-test requires twelve unique people in two groups of six")
    for case in cases:
        if (len(case.sessions) != 8 or tuple(visit.number for visit in case.sessions) != tuple(range(1, 9))
                or tuple(visit.virtual_day for visit in case.sessions) != DAY_OFFSETS
                or any(len(visit.utterances) != 10 or len(visit.synthetic_assistant) != 10
                       for visit in case.sessions)):
            raise ValueError("C-test v1 requires the fixed 8 x 10 script and virtual calendar")
    data = [asdict(case) for case in cases]
    raw = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return {"version": VERSION, "fictional_only": True, "frozen": False,
            "virtual_clock_only": True, "virtual_epoch": VIRTUAL_EPOCH.isoformat(),
            "synthetic_assistant_context": True, "online_model_reply_count": 0,
            "case_count": len(cases), "sessions": 96, "user_turns": 960,
            "groups": {"change": 6, "control": 6}, "source_sha256": hashlib.sha256(raw).hexdigest(),
            "main_rule": {"control_false_triggers": 0, "change_min_triggered": 4,
                          "change_denominator": 6, "control_denominator": 6},
            "coverage_limit": "two_day_functional_impact; not three_day_post_baseline_growth",
            "cases": data}
