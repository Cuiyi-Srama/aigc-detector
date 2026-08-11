#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AIGC 文本检测器 v8 (Python 跨平台版)
====================================
基于统计特征 + 可选 LLM 深度检测的 AI 生成内容检测工具。
纯本地离线运行，支持中文/英文，17维检测 + 6大模型指纹 + 论证脚手架。

平台: Linux / Windows / macOS / Android(Termux)
用法:
  python3 aigc_detector.py -f file.txt           # 检测文件
  python3 aigc_detector.py -t "文本内容"          # 直接检测文本
  echo "文本" | python3 aigc_detector.py          # 管道输入
  python3 aigc_detector.py -f file.txt --json     # JSON 输出(便于集成)
  python3 aigc_detector.py --demo                 # 内置示例演示
  # ---- v8 LLM Pro 引擎 (可选) ----
  python3 aigc_detector.py --list-models          # 查看可选 GGUF 模型清单
  python3 aigc_detector.py --download-model lmstudio-community/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q4_K_M.gguf  # 下载模型
  python3 aigc_detector.py -f file.txt --model /path/to/model.gguf   # 启用 LLM 深度检测

v8 新增 (相对 v7):
  1. LLM Pro 引擎: 加载本地 GGUF 模型, 计算真 Perplexity(困惑度) + top-1 可预测率
     + 目标 token 排名百分位 (GPTZero 核心原理的本地实现, 基于 llama.cpp)
  2. --model 指定 GGUF 路径, 无 llama-cpp-python 时自动降级为 17 维启发式
  3. --list-models / --download-model: HuggingFace 模型下载支持 (默认 hf-mirror 镜像)
  4. LLM 结果与 17 维启发式融合: 最终概率 = 0.6*启发式 + 0.4*LLM深度分
"""

import re
import sys
import math
import json
import argparse

VERSION = "8.0.0"

# ============ v8: HuggingFace 模型清单 (LLM Pro 引擎) ============
# repo: 仓库 | file: 文件名 | size: 大小 | desc: 说明 | lang: 适用语言 | verified: 已验证
HF_MODELS = [
    {'repo': 'lmstudio-community/gemma-4-E4B-it-GGUF', 'file': 'gemma-4-E4B-it-Q4_K_M.gguf', 'size': '5.3GB', 'desc': 'Gemma4 4B 高效版 Q4_K_M (推荐, 平衡)', 'lang': '中英', 'verified': True},
    {'repo': 'lmstudio-community/gemma-4-E4B-it-GGUF', 'file': 'gemma-4-E4B-it-Q6_K.gguf', 'size': '6.2GB', 'desc': 'Gemma4 4B Q6_K 更高精度', 'lang': '中英', 'verified': True},
    {'repo': 'lmstudio-community/gemma-4-E4B-it-GGUF', 'file': 'gemma-4-E4B-it-Q8_0.gguf', 'size': '8.0GB', 'desc': 'Gemma4 4B Q8_0 接近无损精度', 'lang': '中英', 'verified': True},
    {'repo': 'Qwen/Qwen2.5-1.5B-Instruct-GGUF', 'file': 'qwen2.5-1.5b-instruct-q4_k_m.gguf', 'size': '1.0GB', 'desc': '通义千问2.5 1.5B 轻量 (中文优秀)', 'lang': '中文', 'verified': False},
    {'repo': 'Qwen/Qwen2.5-3B-Instruct-GGUF', 'file': 'qwen2.5-3b-instruct-q4_k_m.gguf', 'size': '1.9GB', 'desc': '通义千问2.5 3B (中文更强)', 'lang': '中文', 'verified': False},
    {'repo': 'bartowski/Llama-3.2-1B-Instruct-GGUF', 'file': 'Llama-3.2-1B-Instruct-Q4_K_M.gguf', 'size': '0.9GB', 'desc': 'Meta Llama3.2 1B 轻量 (英文)', 'lang': '英文', 'verified': False},
    {'repo': 'bartowski/Llama-3.2-3B-Instruct-GGUF', 'file': 'Llama-3.2-3B-Instruct-Q4_K_M.gguf', 'size': '2.0GB', 'desc': 'Meta Llama3.2 3B 均衡 (英文)', 'lang': '英文', 'verified': False},
]
HF_MIRROR = 'https://hf-mirror.com'   # 国内直连镜像 (hf-mirror.com)
HF_OFFICIAL = 'https://huggingface.co'

# ============ 中文模式 (v6 继承) ============
AI_PATTERNS_CN = ['首先','其次','然后','最后','此外','另外','同时','与此同时','不仅如此','值得注意的是','需要指出','需要强调的是','综上所述','总之','总的来看','从整体来看','然而','因此','所以','导致','进而','从而','一方面','另一方面','换句话说','正因如此','就像前面说的','不是.*而是','从来不是','注定是','就能','即可','便能','定能','必将','无需','只需','切勿','始终','永远','绝对','完全','从根源上','从根本','核心是','关键在于','本质是','归根结底','从根本上','在.*基础上','以.*为核心']

# ============ 英文模板 (v7 新增) ============
AI_PATTERNS_EN = [
    r'\bMoreover\b', r'\bFurthermore\b', r'\bAdditionally\b', r'\bIn addition\b',
    r'\bHowever\b', r'\bNevertheless\b', r'\bNonetheless\b',
    r"\bIn today's (fast-paced|rapidly|digital)? ?world\b",
    r'\bIt is important to note\b', r"\bIt's important to note\b",
    r'\bIt is worth (noting|mentioning)\b',
    r'\bIn conclusion\b', r'\bTo summarize\b', r'\bTo sum up\b',
    r'\bIn summary\b', r'\bOverall\b',
    r'\b(delve|unlock|harness|leverage|utilize|facilitate|optimize)\b',
    r'\b(crucial|essential|vital|paramount|pivotal)\b',
    r'\b(various|numerous|myriad|plethora|multitude)\b',
    r'\bgame-?changer\b', r'\bparadigm shift\b',
    r'\bat the end of the day\b', r'\bone might (argue|say)\b',
    r'\b(undoubtedly|undeniably|unquestionably)\b',
    r'\bnot only\b.*\bbut also\b', r'\bwhether\b.*\bor not\b',
    r'\bin order to\b', r'\bdue to the fact that\b',
    r'\ba wide range of\b', r'\bin the realm of\b',
    r'\bwhen it comes to\b', r'\bit should be noted\b',
    r'\bplays? a (crucial|vital|significant|key) role\b',
    r'\bimportantly\b', r'\bcrucially\b', r'\bsignificantly\b',
]

TONE_WORDS_CN = ['啊','呢','吧','嘛','呗','啦','哦','噢','咦','哇','嘿','哎','唉','嗯','哈','呀']
TONE_WORDS_EN = [r'\boh\b', r'\bwow\b', r'\bhey\b', r'\bwell\b', r'\bum\b', r'\buh\b',
                 r'\bhmm\b', r'\byeah\b', r'\bnah\b', r'\bgosh\b', r'\boops\b']  # v7

SUBJECTIVE_WORDS_CN = ['我觉得','我认为','我感觉','我个人','说实话','说真的','老实说','没想到','突然发现','越来越觉得','搞不懂']
SUBJECTIVE_WORDS_EN = [r'\bI think\b', r'\bI feel\b', r'\bI believe\b', r'\bin my opinion\b',
                       r'\bto me\b', r'\bpersonally\b', r'\bhonestly\b', r'\bfrom my perspective\b',
                       r'\bI guess\b', r'\bI suppose\b']  # v7

ABSTRACT_WORDS_CN = ['逻辑','规律','核心','原则','思维','意识','方式','方法','能力','技巧','策略','体系','框架','结构','机制','模式','层面','维度','角度','领域','方面','环节','步骤','流程','标准','准则','规范','底层','顶层','根源','本质','导向','认知','范式','全局观','心法','方法论','闭环','抓手','赋能','赛道','颗粒度','对齐','漏斗','壁垒','阈值','纵深']

DS_PATTERNS_CN = ['我们不说空话','直接进入正题','我们接着','重新认识你的对手','颠覆常规','把.*彻底看透','背后的.*潜规则','最让人头疼的','终极底牌','改变一个认知','本质上不是.*而是','从来不是','记住口诀','只推一步','不多想','首尾呼应法','倒叙阅读法','定点清除','逻辑顺滑测试','最小差异法','先易后难','前后夹击','三步走','一场.*博弈战','心理博弈','信息检索战','纸老虎','牵着鼻子走','默契游戏','一场.*信心战','螺旋式上升','第一步','第二步','第三步','第四步','第五步','第一层','第二层','第三层','第四层','第一线索','第二线索','第三线索','三个核心线索','完形全局观','先破误区','建立你的','底层逻辑','彻底理顺','根深蒂固的认知','连根拔除','鹰眼','毛线团','拼图','阶梯','灯塔','祝你下笔有神','下笔有神','去实战中检验','带着.*去实战','祝你的每一笔','去考场上一试身手','带着这份从容','当.*时.*却']
DB_PATTERNS_CN = ['总体来说','总的来说','在.*过程中','通过.*方式','需要.*注意','应该.*考虑','可以.*看到','从.*角度','对于.*来说','在.*方面','具有.*特点','最直接','最不绕弯子','一针见血','不废话','平铺直叙','条理清晰','毫无灵魂','像.*PPT','结构规整','用词平衡','情感平滑']
WX_PATTERNS_CN = ['在.*的长河中','深刻地','显著地','极大地','不可或缺','至关重要','举足轻重','具有重要的','随着.*的发展','在.*背景下','引起了广泛','伴随着.*的浪潮','在.*的新时代','全面.*把握','系统.*阐述','深刻.*揭示','充分.*体现','从.*角度出发','立足于.*实际']
KIMI_PATTERNS_CN = ['超长上下文','长文总结','文档解析','多文档对比','无损上下文','万字以上','整本.*材料','我来帮你.*分析','让我为你.*总结','核心要点如下','梳理出.*关键']
TY_PATTERNS_CN = ['逻辑严密','层层递进','环环相扣','条分缕析','基于.*数据','据.*统计','根据.*报告','技术架构','实施方案','落地路径','最佳实践','显著提升','大幅降低','有效改善']
YB_PATTERNS_CN = ['公众号.*爆文','流量密码','读者粘性','打开率','微信生态','私域流量','社群运营','用户画像','品牌调性','内容矩阵','种草.*拔草','转化链路']

# ============ 论证脚手架 ============
SCAFFOLD_STAGES = {
    'debunk': [r'很多.*误区', r'最.*误区', r'先破误区', r'第一个就是', r'依赖语感', r'见空就填', r'走一步看一步', r'只查语法',
               r'common misconception', r'myth', r'misconception', r'contrary to popular'],
    'build': [r'要破解', r'建立.*的', r'核心.*是', r'三步走', r'底层逻辑', r'彻底.*理顺', r'全局观', r'心法', r'不是.*而是', r'重新认识',
              r'framework', r'step-?by-?step', r'the key is', r'first principle'],
    'dissect': [r'第一线索', r'第二线索', r'第三线索', r'三个.*核心', r'词汇复现', r'逻辑信号', r'情感.*基调', r'归纳为',
                r'analyze', r'break down', r'three (key|main|core)'],
    'practice': [r'记住', r'口诀', r'终极', r'底牌', r'最小差异法', r'逻辑.*测试', r'不要.*恋战', r'代入.*原文',
                 r'practice', r'apply it', r'try it yourself', r'memorize'],
    'elevate': [r'说到底', r'就是一场', r'祝你的', r'默契.*游戏', r'你会发现', r'带着.*去', r'不再.*而是',
                r'you will (find|see)', r'at the end of the day', r'ultimately'],
}

POS_WORDS_CN = ['好','棒','赞','喜欢','开心','快乐','幸福','美好','优秀','精彩','厉害','不错','满意','舒服','温暖','感动','惊喜','兴奋']
NEG_WORDS_CN = ['坏','差','烂','讨厌','伤心','痛苦','难过','糟糕','失败','失望','生气','愤怒','害怕','担心','焦虑','孤独','悲哀']
POS_WORDS_EN = [r'\bgood\b', r'\bgreat\b', r'\bnice\b', r'\blike\b', r'\bhappy\b', r'\bjoy\b', r'\bwonderful\b',
                r'\bexcellent\b', r'\bamazing\b', r'\blove\b', r'\bbeautiful\b', r'\bwarm\b', r'\btouching\b',
                r'\bsurprising\b', r'\bexcited\b', r'\bfair\b', r'\bfree\b', r'\bbetter\b', r'\bfavorite\b']
NEG_WORDS_EN = [r'\bbad\b', r'\bterrible\b', r'\bawful\b', r'\bhate\b', r'\bsad\b', r'\bpainful\b', r'\bmiserable\b',
                r'\bfail\b', r'\bdisappointed\b', r'\bangry\b', r'\bafraid\b', r'\bworried\b', r'\banxious\b',
                r'\blonely\b', r'\bcrowded\b', r'\bunsafe\b', r'\bworse\b', r'\bproblem\b', r'\bissue\b']

# ============ 工具函数 ============
def is_chinese_char(c):
    cp = ord(c)
    return (0x4e00 <= cp <= 0x9fff) or (0x3400 <= cp <= 0x4dbf)

def has_chinese(text):
    return any(is_chinese_char(c) for c in text[:2000])

def tokenize(text, cn):
    if cn:
        return [w for w in re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z0-9]+', text) if w]
    return re.findall(r"[a-zA-Z0-9]+(?:'[a-zA-Z]+)?", text.lower())

def split_sentences(text, cn):
    if cn:
        t = re.sub(r'[。！？；]', lambda m: {'。': '.', '！': '!', '？': '?', '；': ';'}[m.group(0)], text)
    else:
        t = text
    return [s.strip() for s in re.split(r'[.!?\n;]+', t) if len(s.strip()) > 3]

def split_paragraphs(text):
    return [p.strip() for p in re.split(r'\n{2,}', text) if len(p.strip()) > 10]

def calc_burstiness(arr):
    if len(arr) < 2:
        return 0
    lens = [len(x) if isinstance(x, str) else x for x in arr]
    avg = sum(lens) / len(lens)
    std = math.sqrt(sum((l - avg) ** 2 for l in lens) / len(lens))
    return std / avg if avg > 0 else 0

def calc_hapax_ratio(words):
    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    hapax = sum(1 for c in freq.values() if c == 1)
    return hapax / len(freq) if freq else 0

def count_matches(text, patterns):
    cnt = 0
    for pat in patterns:
        try:
            cnt += len(re.findall(pat, text, re.I))
        except Exception:
            pass
    return cnt

def clamp(v):
    return max(0, min(100, v))

# ============ 核心分析 ============
def analyze(text):
    text = text.strip()
    if len(text) < 100:
        return {'error': '文本至少需要100个字符 (至少100字符, 越长越准)'}
    cn = has_chinese(text)
    lang = 'zh' if cn else 'en'
    words = tokenize(text, cn)
    if len(words) < 10:
        return {'error': '词汇量太少'}

    sentences = split_sentences(text, cn)
    paragraphs = split_paragraphs(text)

    burstiness = calc_burstiness(sentences)
    unique = set(words)
    ttr = len(unique) / len(words)
    hapax_ratio = calc_hapax_ratio(words)

    patterns = AI_PATTERNS_CN if cn else AI_PATTERNS_EN
    template_count = count_matches(text, patterns)
    template_density = template_count / len(words)

    tone_pats = TONE_WORDS_CN if cn else TONE_WORDS_EN
    tone_count = count_matches(text, tone_pats)
    tone_density = tone_count / len(words)

    subj_pats = SUBJECTIVE_WORDS_CN if cn else SUBJECTIVE_WORDS_EN
    subj_count = count_matches(text, subj_pats)
    subj_density = subj_count / len(words)

    de_count = count_matches(text, [r'的'])
    de_density = de_count / len(words)

    abstract_count = count_matches(text, ABSTRACT_WORDS_CN) if cn else 0
    abstract_ratio = abstract_count / len(words)

    punct_list = ['，', '。', '！', '？', '；', '：', '、', '……', '——'] if cn else [',', '.', '!', '?', ';', ':', '-']
    punct_types = sum(1 for p in punct_list if p in text)

    complex_cnt = 0
    for s in sentences:
        commas = len(re.findall(r'[，,；;]', s))
        conns = len(re.findall(r'虽然|但是|因为|所以|如果|尽管|然而|因此|而且|并且|不仅|与其|不如|除非|无论|不管' if cn
                               else r'although|but|because|so|if|though|however|therefore|and|also|not only|unless|whether|while',
                               s))
        if commas >= 2 or conns >= 1:
            complex_cnt += 1
    complex_ratio = complex_cnt / len(sentences) if sentences else 0

    para_burstiness = calc_burstiness(paragraphs) if len(paragraphs) >= 2 else 0

    pos_words = POS_WORDS_CN if cn else POS_WORDS_EN
    neg_words = NEG_WORDS_CN if cn else NEG_WORDS_EN
    emotion_density = (count_matches(text, pos_words) + count_matches(text, neg_words)) / len(words)

    section_nums = count_matches(text, [r'[一二三四五六七八九十]、']) if cn else count_matches(text, [r'\b(First|Second|Third|Fourth|Fifth)\b'])
    section_density = section_nums / len(words)

    qt = count_matches(text, [r'[""][^""]{2,20}[""]'])
    quoted_density = qt / len(words)

    metaphor_pats = [r'像.*拼图', r'像.*阶梯', r'像.*灯塔', r'纸老虎', r'毛线团', r'鹰眼', r'就像.*一样', r'如同.*一般',
                     r'是一场.*战', r'是一场.*游戏', r'是.*的阶梯', r'是.*的钥匙'] if cn else [
        r'\b(like|as) (a )?(puzzle|ladder|beacon|light|game|battle|war|key|map)\b',
        r'\b(a|the) (puzzle|ladder|beacon|light) of\b']
    metaphor_count = count_matches(text, metaphor_pats)
    metaphor_density = metaphor_count / len(words)

    bracket_pairs = count_matches(text, [r'\([^)]{3,30}\)', r'（[^）]{3,30}）', r'——[^—]{3,30}——'])
    bracket_density = bracket_pairs / len(words)

    # 论证脚手架
    stage_count = 0
    stage_hits = 0
    for name, pats in SCAFFOLD_STAGES.items():
        hit = False
        for p in (paragraphs if paragraphs else [text]):
            for pat in pats:
                if re.search(pat, p, re.I):
                    hit = True
                    break
            if hit:
                break
        if hit:
            stage_count += 1
            stage_hits += 1
    scaffold_density = stage_hits / max(1, len(paragraphs))

    # 模型指纹
    flags = []
    model_bonus = 0
    ds_score = count_matches(text, DS_PATTERNS_CN)
    if ds_score >= 5 or section_nums >= 4:
        flags.append('DS:分节%d+指纹%d+脚手架%d/5' % (section_nums, ds_score, stage_count))
        model_bonus += 22
        if stage_count >= 4:
            model_bonus += 8
    elif ds_score >= 3:
        flags.append('疑似DS:指纹%d' % ds_score)
        model_bonus += 12

    db_score = count_matches(text, DB_PATTERNS_CN)
    if db_score >= 5:
        flags.append('豆包:%d处直白模板' % db_score)
        model_bonus += 12
    elif db_score >= 3:
        flags.append('疑似豆包:%d' % db_score)
        model_bonus += 6

    wx_score = count_matches(text, WX_PATTERNS_CN)
    if wx_score >= 4:
        flags.append('文心:%d处华丽修饰' % wx_score)
        model_bonus += 10

    kimi_score = count_matches(text, KIMI_PATTERNS_CN)
    if kimi_score >= 4:
        flags.append('Kimi:%d处长文特征' % kimi_score)
        model_bonus += 8

    ty_score = count_matches(text, TY_PATTERNS_CN)
    if ty_score >= 4:
        flags.append('通义:%d处逻辑特征' % ty_score)
        model_bonus += 8

    yb_score = count_matches(text, YB_PATTERNS_CN)
    if yb_score >= 4:
        flags.append('元宝:%d处运营特征' % yb_score)
        model_bonus += 8

    if section_nums >= 3:
        flags.append('数字分节%d处' % section_nums)
    if stage_count >= 4:
        flags.append('完整论证脚手架%d/5阶段' % stage_count)
    elif stage_count >= 3:
        flags.append('论证脚手架%d/5阶段' % stage_count)
    if tone_count == 0:
        flags.append('零语气词' if cn else 'zero tone words')
    if de_density > 0.06 and cn:
        flags.append('的字密度%.1f%%' % (de_density * 100))
    if abstract_ratio > 0.03 and cn:
        flags.append('抽象堆砌%.1f%%' % (abstract_ratio * 100))
    if bracket_pairs >= 3:
        flags.append('括号嵌套%d处' % bracket_pairs)
    if metaphor_count >= 3:
        flags.append('功能比喻%d处' % metaphor_count)

    scores = {
        'burstiness': {'raw': burstiness, 'ai': clamp((1 - burstiness / 0.55) * 100), 'w': 12, 'n': '句子突发性' if cn else 'Burstiness'},
        'ttr': {'raw': ttr, 'ai': clamp((1 - ttr / 0.6) * 100), 'w': 7, 'n': '词汇多样性' if cn else 'TTR'},
        'hapax': {'raw': hapax_ratio, 'ai': clamp((1 - hapax_ratio / 0.65) * 100), 'w': 8, 'n': '罕见词比例' if cn else 'Hapax'},
        'template': {'raw': template_density, 'ai': clamp(template_density / 0.06 * 100), 'w': 14, 'n': '模板套话' if cn else 'Templates'},
        'tone': {'raw': tone_density, 'ai': clamp((1 - tone_density / 0.012) * 100), 'w': 9, 'n': '语气词缺失' if cn else 'Tone words'},
        'subj': {'raw': subj_density, 'ai': clamp((1 - subj_density / 0.008) * 100), 'w': 5, 'n': '主观表达' if cn else 'Subjectivity'},
        'de': {'raw': de_density, 'ai': clamp((de_density - 0.03) / 0.05 * 100) if cn else 0, 'w': 7, 'n': '的密度' if cn else 'de-density'},
        'abstract': {'raw': abstract_ratio, 'ai': clamp(abstract_ratio / 0.04 * 100), 'w': 6, 'n': '抽象概念' if cn else 'Abstract'},
        'punct': {'raw': punct_types, 'ai': clamp((1 - punct_types / 9) * 100), 'w': 4, 'n': '标点多样性' if cn else 'Punctuation'},
        'complex': {'raw': complex_ratio, 'ai': clamp((1 - complex_ratio / 0.6) * 100), 'w': 5, 'n': '句式复杂度' if cn else 'Complexity'},
        'para': {'raw': para_burstiness, 'ai': clamp((1 - para_burstiness / 0.55) * 100), 'w': 5, 'n': '段落均匀度' if cn else 'Paragraphs'},
        'emotion': {'raw': emotion_density, 'ai': clamp((1 - emotion_density / 0.035) * 100), 'w': 4, 'n': '情感表达' if cn else 'Emotion'},
        'section': {'raw': section_density, 'ai': clamp(section_density / 0.015 * 100), 'w': 8, 'n': '数字分节' if cn else 'Sections'},
        'quoted': {'raw': quoted_density, 'ai': clamp(quoted_density / 0.02 * 100), 'w': 6, 'n': '引号术语' if cn else 'Quotes'},
        'metaphor': {'raw': metaphor_density, 'ai': clamp(metaphor_density / 0.015 * 100), 'w': 5, 'n': '比喻密度' if cn else 'Metaphors'},
        'scaffold': {'raw': scaffold_density, 'ai': clamp(stage_count / 5 * 100), 'w': 5, 'n': '论证脚手架' if cn else 'Scaffold'},
        'bracket': {'raw': bracket_density, 'ai': clamp(bracket_density / 0.008 * 100), 'w': 4, 'n': '括号嵌套' if cn else 'Brackets'},
    }

    tw = 0
    ws = 0
    details = []
    for key, s in scores.items():
        ws += s['ai'] * s['w']
        tw += s['w']
        details.append({'key': key, 'name': s['n'], 'ai_score': round(s['ai'], 1), 'raw': round(s['raw'], 4), 'weight': s['w']})

    ai_probability = round(ws / tw * 10) / 10 if tw > 0 else 50
    ai_probability = min(100, ai_probability + model_bonus * 0.5)

    if ai_probability < 25:
        level = 'Human'
    elif ai_probability < 40:
        level = 'Maybe AI'
    elif ai_probability < 60:
        level = 'Uncertain'
    elif ai_probability < 80:
        level = 'Likely AI'
    else:
        level = 'Very Likely AI'

    return {
        'ai_probability': ai_probability,
        'level': level,
        'language': lang,
        'details': sorted(details, key=lambda d: -d['ai_score']),
        'flags': flags,
        'stats': {
            'chars': len(text), 'words': len(words), 'unique': len(unique),
            'sentences': len(sentences), 'paragraphs': len(paragraphs),
            'tone_words': tone_count, 'templates': template_count,
            'section_nums': section_nums, 'quoted_terms': qt,
            'metaphors': metaphor_count, 'brackets': bracket_pairs,
            'scaffold_stages': stage_count,
        },
    }

# ============ v8: LLM Pro 引擎 (可选, 需 llama-cpp-python) ============
# 原理 (GPTZero 核心的本地实现):
#   AI 文本 -> 低困惑度(模型"意料之中") + 高 top-1 可预测率 + 目标token排名靠前
#   人类文本 -> 高困惑度 + 低可预测率 + 目标token排名分散
def analyze_llm(text, model_path, max_tokens=1500):
    """使用本地 GGUF 模型深度检测。返回困惑度/可预测率/排名等指标。"""
    try:
        from llama_cpp import Llama
    except ImportError:
        return {'error': '需要安装 llama-cpp-python: pip install llama-cpp-python (加载模型 %s)' % model_path}
    import os
    import numpy as np

    try:
        llm = Llama(model_path=model_path, n_ctx=2048,
                    n_threads=max(1, (os.cpu_count() or 4) // 2), verbose=False)
    except Exception as e:
        return {'error': '模型加载失败: %s' % e}

    try:
        tokens = llm.tokenize(text.encode('utf-8'))
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
        if len(tokens) < 20:
            return {'error': '文本太短, Pro引擎至少需要20个token'}
        llm.reset()
        llm.eval(tokens)
        scores = llm.scores
        n = min(len(scores), len(tokens) - 1)
        if n <= 0:
            return {'error': '模型推理未产生有效输出'}
        pred_hits = 0
        rank_pcts = []
        nll_sum = 0.0
        for i in range(n):
            logits = np.asarray(scores[i], dtype=np.float32)
            target = tokens[i + 1]
            if target >= len(logits):
                continue
            pred = int(np.argmax(logits))
            if pred == target:
                pred_hits += 1
            rank = int((logits > logits[target]).sum())
            rank_pcts.append(rank / len(logits))
            # 近似 logsumexp (top-512 截断, 误差 < 0.01 nats)
            topk = np.partition(logits, -512)[-512:]
            mx = float(topk[-1])
            lse = mx + math.log(float(np.exp(topk - mx).sum()))
            nll_sum += (lse - float(logits[target]))
        del llm
        pred_rate = pred_hits / n
        rank_med = sorted(rank_pcts)[len(rank_pcts) // 2] if rank_pcts else 0.5
        ppl = math.exp(nll_sum / n)
        # 融合: AI 文本 -> pred_rate 高 + rank_med 低
        s_pred = min(1.0, pred_rate / 0.35)
        s_rank = min(1.0, max(0.0, (0.25 - rank_med) / 0.13))
        llm_score = clamp(100 * (0.55 * s_pred + 0.45 * s_rank))
        return {
            'ppl': round(ppl, 1),
            'pred_rate': round(pred_rate, 3),
            'rank_pct': round(rank_med, 3),
            'llm_score': round(llm_score, 1),
            'tokens': n,
        }
    except Exception as e:
        return {'error': '推理失败: %s' % e}


def download_model(spec, hf_base=HF_MIRROR, dest='.'):
    """从 HuggingFace (默认 hf-mirror 镜像) 下载 GGUF 模型。spec: repo/file 或 repo:file"""
    import os
    import urllib.request
    spec = spec.strip()
    if spec.count('/') == 2:            # org/repo/file
        repo, fname = spec.rsplit('/', 1)
    elif spec.count('/') == 1:          # repo/file
        repo, fname = spec.split('/', 1)
    else:
        return '格式错误: 应为 org/repo/file.gguf 或 org/repo:file.gguf'
    url = '%s/%s/resolve/main/%s' % (hf_base.rstrip('/'), repo, fname)
    out = os.path.join(dest, os.path.basename(fname))
    if os.path.exists(out) and os.path.getsize(out) > 1024 * 1024:
        return '已存在: %s (跳过下载)' % out

    def _hook(blocks, block_size, total):
        done = blocks * block_size
        pct = min(100, done * 100 // total) if total > 0 else 0
        print('\r  下载中... %d%% (%d/%d MB)' % (pct, done // 1048576, total // 1048576), end='', flush=True)

    print('下载: %s' % url)
    try:
        urllib.request.urlretrieve(url, out, _hook)
        print('\n完成: %s' % out)
        return out
    except Exception as e:
        if os.path.exists(out):
            os.remove(out)
        return '下载失败: %s' % e


def list_models():
    lines = []
    lines.append('=' * 78)
    lines.append('  可选 GGUF 模型 (HuggingFace)  —  LLM Pro 引擎')
    lines.append('=' * 78)
    lines.append('%-46s %8s  %s' % ('模型', '大小', '说明'))
    lines.append('-' * 78)
    for m in HF_MODELS:
        v = ' [已验证]' if m.get('verified') else ''
        lines.append('%-46s %8s  %s%s' % (m['file'], m['size'], m['desc'], v))
    lines.append('-' * 78)
    lines.append('下载: python3 aigc_detector.py --download-model org/repo/file.gguf')
    lines.append('镜像: 默认 hf-mirror.com (国内直连), 可用 --hf-base https://huggingface.co 切换官方源')
    lines.append('示例: python3 aigc_detector.py --download-model lmstudio-community/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q4_K_M.gguf')
    lines.append('=' * 78)
    return '\n'.join(lines)


def level_of(p):
    if p < 25:
        return 'Human'
    if p < 40:
        return 'Maybe AI'
    if p < 60:
        return 'Uncertain'
    if p < 80:
        return 'Likely AI'
    return 'Very Likely AI'

# ============ 输出 ============
def render_text(r, verbose=True):
    lines = []
    lines.append('=' * 46)
    lines.append('  AIGC 文本检测器 v%s (%s)' % (VERSION, '中文' if r['language'] == 'zh' else 'English'))
    lines.append('=' * 46)
    p = r['ai_probability']
    bar_len = int(p / 5)
    lines.append('  AI概率: %.1f%%  %s%s 判定: %s' % (p, '#' * bar_len, '-' * (20 - bar_len), r['level']))
    if 'llm' in r:
        ll = r['llm']
        lines.append('  [Pro] LLM深度: 困惑度=%.1f 可预测率=%.1f%% 排名=%.1f%% (LLM分 %.1f)' %
                     (ll.get('ppl', 0), ll.get('pred_rate', 0) * 100, ll.get('rank_pct', 0) * 100, ll.get('llm_score', 0)))
    lines.append('  文本: %d字符 / %d词 / %d句 / %d段' % (r['stats']['chars'], r['stats']['words'], r['stats']['sentences'], r['stats']['paragraphs']))
    if r['flags']:
        lines.append('  特征: %s' % '; '.join(r['flags']))
    if verbose:
        lines.append('-' * 46)
        for d in r['details']:
            bar = '#' * int(d['ai_score'] / 5)
            lines.append('  %-8s %6.1f (w%d) %s' % (d['name'], d['ai_score'], d['weight'], bar))
    lines.append('=' * 46)
    lines.append('  结果基于简易统计算法，仅供参考')
    return '\n'.join(lines)

# ============ 入口 ============
def main():
    ap = argparse.ArgumentParser(description='AIGC文本检测器 v%s (跨平台 Python 版)' % VERSION)
    ap.add_argument('-f', '--file', help='输入文本文件路径')
    ap.add_argument('-t', '--text', help='直接输入文本')
    ap.add_argument('--dir', help='批量检测目录下所有 .txt/.md/.markdown 文件 (递归)')
    ap.add_argument('--json', action='store_true', help='JSON 格式输出')
    ap.add_argument('--quiet', action='store_true', help='精简输出(仅概率+判定)')
    ap.add_argument('--demo', action='store_true', help='运行内置示例')
    ap.add_argument('--version', action='version', version='aigc-detector v' + VERSION)
    # v8: LLM Pro 引擎
    ap.add_argument('--model', metavar='GGUF', help='本地 GGUF 模型路径, 启用 LLM 深度检测 (需 pip install llama-cpp-python)')
    ap.add_argument('--list-models', action='store_true', help='列出可选 GGUF 模型 (HuggingFace)')
    ap.add_argument('--download-model', metavar='SPEC', help='从 HuggingFace 下载模型, 格式: org/repo/file.gguf')
    ap.add_argument('--hf-base', default=HF_MIRROR, help='HuggingFace 源 (默认 hf-mirror.com 镜像, 可换 https://huggingface.co)')
    ap.add_argument('--model-dir', default='.', help='--download-model 的保存目录 (默认当前目录)')
    args = ap.parse_args()

    if args.list_models:
        print(list_models())
        return

    if args.download_model:
        msg = download_model(args.download_model, args.hf_base, args.model_dir)
        print(msg)
        return

    samples = {
        'human': '我与父亲不相见已二年余了，我最不能忘记的是他的背影。那年冬天，祖母死了，父亲的差使也交卸了，正是祸不单行的日子，我从北京到徐州，打算跟着父亲奔丧回家。到徐州见着父亲，看见满院狼藉的东西，又想起祖母，不禁簌簌地流下眼泪。父亲说："事已如此，不必难过，好在天无绝人之路！"回家变卖典质，父亲还了亏空；又借钱办了丧事。这些日子，家中光景很是惨淡，一半为了丧事，一半为了父亲赋闲。丧事完毕，父亲要到南京谋事，我也要回北京念书，我们便同行。',
        'ai': '首先，考生需要明确一个核心原则：所有的正确答案均来源于原文的同义替换。命题人遵循固定的出题套路，所有干扰项都可以通过"无中生有""偷换概念""以偏概全"三大陷阱进行分类识别。其次，考场上必须采用"先题后文、精准定位"的答题顺序，这是经过大量实证研究验证的最优策略。考生在拿到试卷后，应当先用30秒快速浏览题目和选项，圈画出关键词，然后回到文章中按照关键词进行定位。在此过程中，需要特别注意转折词、因果词、举例词等逻辑信号，这些信号词往往直接指向正确答案所在的位置。此外，针对主旨大意题，考生应重点关注文章首段、尾段以及各段首句，这些位置通常包含文章的核心论点。',
    }

    if args.demo:
        for name, txt in samples.items():
            r = analyze(txt)
            print('## 示例: %s' % name)
            print(render_text(r))
            print()
        return

    if args.dir:
        import glob as _glob
        import os as _os
        files = []
        for ext in ('*.txt', '*.md', '*.markdown'):
            files.extend(_glob.glob(_os.path.join(args.dir, '**', ext), recursive=True))
        if not files:
            print('目录中未找到 .txt/.md/.markdown 文件: %s' % args.dir, file=sys.stderr)
            sys.exit(1)
        results = []
        for f in sorted(files):
            try:
                with open(f, 'r', encoding='utf-8', errors='ignore') as fh:
                    r = analyze(fh.read())
                if 'error' in r:
                    results.append((f, None, r['error']))
                else:
                    results.append((f, r['ai_probability'], r['level']))
            except Exception as e:
                results.append((f, None, str(e)))
        if args.json:
            print(json.dumps([{'file': f, 'ai_probability': p, 'level': l, 'error': e}
                              for f, p, l, e in [(a, b, c, '') for a, b, c in results]
                              if True], ensure_ascii=False, indent=2))
        else:
            print('%-40s %8s  %s' % ('文件', 'AI概率', '判定'))
            print('-' * 70)
            for f, p, l in results:
                if p is None:
                    print('%-40s %8s  %s' % (f, 'ERR', l))
                else:
                    print('%-40s %6.1f%%  %s' % (f, p, l))
        return

    if args.file:
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception as e:
            print('读取文件失败: %s' % e, file=sys.stderr)
            sys.exit(1)
    elif args.text:
        text = args.text
    else:
        text = sys.stdin.read()

    if not text.strip():
        print('未检测到输入。用法: python3 aigc_detector.py -f file.txt 或 -t "文本"', file=sys.stderr)
        sys.exit(1)

    r = analyze(text)
    if 'error' in r:
        print('错误: %s' % r['error'], file=sys.stderr)
        sys.exit(1)

    # v8: LLM Pro 引擎融合 (可选)
    if args.model:
        print('加载 LLM 模型: %s ...' % args.model, file=sys.stderr)
        llm_res = analyze_llm(text, args.model)
        if 'error' in llm_res:
            print('警告: %s (已降级为 17 维启发式)' % llm_res['error'], file=sys.stderr)
        else:
            r['llm'] = llm_res
            r['ai_probability'] = round(0.6 * r['ai_probability'] + 0.4 * llm_res['llm_score'], 1)
            r['level'] = level_of(r['ai_probability'])

    if args.json:
        print(json.dumps(r, ensure_ascii=False, indent=2))
    elif args.quiet:
        print('%.1f%% %s' % (r['ai_probability'], r['level']))
    else:
        print(render_text(r))

if __name__ == '__main__':
    main()
