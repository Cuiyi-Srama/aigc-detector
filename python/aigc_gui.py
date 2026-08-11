#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AIGC 文本检测器 v7 — GUI 桌面版
================================
基于 tkinter 的图形界面，跨平台 (Linux / Windows / macOS)。
零额外依赖 (Python 标准库)。

用法:
  python3 aigc_gui.py

功能:
  - 粘贴/打开文件检测
  - AI概率仪表盘 + 判定等级
  - 17维检测明细条形图
  - 特征标记 + 统计面板
  - 内置示例一键加载
"""

import sys
import os
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox

# 保证可独立运行 (直接运行本文件时也能找到 aigc_detector)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from aigc_detector import analyze, VERSION, analyze_llm, level_of, HF_MODELS
except ImportError:
    # 兜底: 内联引用
    VERSION = "7.0.0"
    from aigc_detector import analyze, VERSION, analyze_llm, level_of, HF_MODELS

LEVEL_COLORS = {
    'Human': '#4caf50',
    'Maybe AI': '#ffc107',
    'Uncertain': '#ff9800',
    'Likely AI': '#f44336',
    'Very Likely AI': '#d32f2f',
}
LEVEL_TEXT = {
    'Human': '人类写作',
    'Maybe AI': '可能含 AI',
    'Uncertain': '不确定',
    'Likely AI': 'AI 生成',
    'Very Likely AI': '极可能 AI 生成',
}

SAMPLES = {
    '人类(背影)': '我与父亲不相见已二年余了，我最不能忘记的是他的背影。那年冬天，祖母死了，父亲的差使也交卸了，正是祸不单行的日子，我从北京到徐州，打算跟着父亲奔丧回家。到徐州见着父亲，看见满院狼藉的东西，又想起祖母，不禁簌簌地流下眼泪。父亲说："事已如此，不必难过，好在天无绝人之路！"回家变卖典质，父亲还了亏空；又借钱办了丧事。这些日子，家中光景很是惨淡，一半为了丧事，一半为了父亲赋闲。丧事完毕，父亲要到南京谋事，我也要回北京念书，我们便同行。',
    '通用AI': '首先，考生需要明确一个核心原则：所有的正确答案均来源于原文的同义替换。命题人遵循固定的出题套路，所有干扰项都可以通过"无中生有""偷换概念""以偏概全"三大陷阱进行分类识别。其次，考场上必须采用"先题后文、精准定位"的答题顺序，这是经过大量实证研究验证的最优策略。考生在拿到试卷后，应当先用30秒快速浏览题目和选项，圈画出关键词，然后回到文章中按照关键词进行定位。在此过程中，需要特别注意转折词、因果词、举例词等逻辑信号，这些信号词往往直接指向正确答案所在的位置。此外，针对主旨大意题，考生应重点关注文章首段、尾段以及各段首句，这些位置通常包含文章的核心论点。',
    'DeepSeek': '我们不说空话，直接进入正题，把高考英语阅读理解这个"纸老虎"彻底拆开来看。一、重新认识你的对手。很多同学对阅读理解有一个根深蒂固的误解：以为考的就是英语水平。但本质上，高考阅读理解不是一场"英语水平测试"，而是一场"信息检索与逻辑推理游戏"。命题老师的任务，是在有限的篇幅内设计出能区分考生思维层级的题目。换句话说，你不需要认识每一个单词，也不需要完全读懂每一句话——你需要的是精准定位信息、排除干扰项、做出逻辑判断的能力。二、颠覆常规的解题顺序。绝大多数同学的做题习惯是从头到尾读完文章再做题。这个方法最大的问题是：你不知道题目问什么，所以你看文章的时候没有焦点。我强烈建议你采用"倒叙阅读法"：第一步，用30秒闪电扫描文章首尾段和各段首句，快速建立文章框架；第二步，直接阅读所有题目和选项，圈出每个问题的"题眼"；第三步，带着这些"题眼"回到文章，像拿着导航地图一样精准定位答案区域。好了，带着这套方法论去实战吧。记住，高考阅读不是考你英语多好，而是考你逻辑多强。祝你下笔有神。',
}


class AigcDetectorApp:
    def __init__(self, root):
        self.root = root
        root.title('AIGC 文本检测器 v%s' % VERSION)
        root.geometry('720x860')
        root.minsize(600, 700)
        root.configure(bg='#12121e')

        self._build_style()
        self._build_header()
        self._build_input()
        self._build_buttons()
        self._build_result()
        self._build_footer()

    def _build_style(self):
        style = ttk.Style()
        try:
            style.theme_use('clam')
        except Exception:
            pass
        style.configure('TFrame', background='#12121e')
        style.configure('TLabel', background='#12121e', foreground='#e0e0e0', font=('Sans', 10))
        style.configure('Title.TLabel', background='#12121e', foreground='#00e5ff',
                        font=('Sans', 18, 'bold'))
        style.configure('Sub.TLabel', background='#12121e', foreground='#888888', font=('Sans', 9))
        style.configure('Accent.TButton', background='#00e5ff', foreground='#000000',
                        font=('Sans', 11, 'bold'), padding=6)
        style.configure('Ghost.TButton', background='#2a2a3e', foreground='#e0e0e0', padding=4)
        style.configure('Detect.TButton', background='#00e5ff', foreground='#000000',
                        font=('Sans', 12, 'bold'), padding=8)
        style.configure('Horizontal.TProgressbar', background='#00e5ff', troughcolor='#2a2a3e',
                        bordercolor='#2a2a3e', lightcolor='#00e5ff', darkcolor='#00e5ff')

    def _build_header(self):
        f = ttk.Frame(self.root)
        f.pack(fill='x', padx=16, pady=(14, 4))
        ttk.Label(f, text='AIGC 文本检测器', style='Title.TLabel').pack()
        ttk.Label(f, text='v%s · 17维检测 · 6大模型指纹 · 100%本地离线 · 中英双语' % VERSION,
                  style='Sub.TLabel').pack()

    def _build_input(self):
        f = ttk.Frame(self.root)
        f.pack(fill='both', expand=True, padx=16, pady=8)
        self.text = scrolledtext.ScrolledText(f, height=10, wrap='word',
                                              bg='#1a1a2e', fg='#e0e0e0',
                                              insertbackground='#00e5ff',
                                              font=('Sans', 11), relief='flat',
                                              padx=10, pady=10)
        self.text.pack(fill='both', expand=True)

    def _build_buttons(self):
        f = ttk.Frame(self.root)
        f.pack(fill='x', padx=16, pady=4)
        ttk.Button(f, text='开始检测', style='Detect.TButton',
                   command=self.detect).pack(side='left', fill='x', expand=True)
        ttk.Button(f, text='清空', style='Ghost.TButton', command=self.clear).pack(side='left', padx=(6, 0))
        ttk.Button(f, text='打开文件', style='Ghost.TButton', command=self.open_file).pack(side='left', padx=(6, 0))

        fs = ttk.Frame(self.root)
        fs.pack(fill='x', padx=16, pady=(4, 0))
        for name in SAMPLES:
            ttk.Button(fs, text=name, style='Ghost.TButton',
                       command=lambda n=name: self.load_sample(n)).pack(side='left', padx=(0, 4))

        # v8: LLM Pro 引擎 (模型选择)
        fm = ttk.Frame(self.root)
        fm.pack(fill='x', padx=16, pady=(4, 0))
        self.model_path = tk.StringVar()
        ttk.Label(fm, text='LLM模型:', background='#12121e', foreground='#00e5ff',
                  font=('Sans', 9)).pack(side='left')
        ttk.Entry(fm, textvariable=self.model_path, font=('Sans', 9)).pack(side='left', fill='x', expand=True, padx=(6, 0))
        ttk.Button(fm, text='选择', style='Ghost.TButton',
                   command=self.choose_model).pack(side='left', padx=(6, 0))
        ttk.Button(fm, text='模型清单', style='Ghost.TButton',
                   command=self.show_models).pack(side='left', padx=(4, 0))
        self.model_hint = ttk.Label(self.root, text='未启用 LLM 深度检测 (17维启发式)',
                                    background='#12121e', foreground='#666666', font=('Sans', 8))
        self.model_hint.pack(fill='x', padx=16)

    def _build_result(self):
        f = ttk.Frame(self.root)
        f.pack(fill='both', expand=True, padx=16, pady=8)

        # 顶部: 分数 + 等级
        top = ttk.Frame(f)
        top.pack(fill='x')
        self.score_label = ttk.Label(top, text='--', font=('Sans', 44, 'bold'),
                                     foreground='#666666')
        self.score_label.pack(side='left', padx=(0, 16))
        self.level_label = ttk.Label(top, text='等待检测', font=('Sans', 18, 'bold'),
                                     foreground='#888888')
        self.level_label.pack(side='left')
        self.flag_label = ttk.Label(top, text='', foreground='#f88', font=('Sans', 9), wraplength=400)
        self.flag_label.pack(side='right', padx=(8, 0))

        # 17维明细滚动区
        self.canvas = tk.Canvas(f, bg='#12121e', highlightthickness=0)
        sb = ttk.Scrollbar(f, orient='vertical', command=self.canvas.yview)
        self.detail_frame = ttk.Frame(self.canvas)
        self.detail_frame.bind('<Configure>',
                               lambda e: self.canvas.configure(scrollregion=self.canvas.bbox('all')))
        self.canvas.create_window((0, 0), window=self.detail_frame, anchor='nw')
        self.canvas.configure(yscrollcommand=sb.set)
        self.canvas.pack(side='left', fill='both', expand=True, pady=(8, 0))
        sb.pack(side='right', fill='y', pady=(8, 0))

        self.detail_rows = []
        self.stats_label = ttk.Label(f, text='', foreground='#00e5ff', font=('Sans', 9))
        self.stats_label.pack(fill='x', pady=(6, 0))

    def _build_footer(self):
        ttk.Label(self.root,
                  text='结果基于简易统计算法生成，仅供参考 · 纯本地运行，文本不会上传',
                  style='Sub.TLabel').pack(side='bottom', pady=8)

    def clear(self):
        self.text.delete('1.0', 'end')
        self.score_label.config(text='--', foreground='#666666')
        self.level_label.config(text='等待检测', foreground='#888888')
        self.flag_label.config(text='')
        self.stats_label.config(text='')
        for row in self.detail_rows:
            row.destroy()
        self.detail_rows = []

    def load_sample(self, name):
        self.clear()
        self.text.insert('1.0', SAMPLES[name])

    def open_file(self):
        path = filedialog.askopenfilename(filetypes=[('文本文件', '*.txt *.md *.html'), ('所有文件', '*.*')])
        if not path:
            return
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                self.clear()
                self.text.insert('1.0', f.read())
        except Exception as e:
            messagebox.showerror('错误', '读取文件失败: %s' % e)

    def choose_model(self):
        path = filedialog.askopenfilename(filetypes=[('GGUF模型', '*.gguf'), ('所有文件', '*.*')],
                                          title='选择本地 GGUF 模型')
        if path:
            self.model_path.set(path)
            self.model_hint.config(text='已启用 LLM 深度检测: %s' % os.path.basename(path),
                                   foreground='#00e5ff')

    def show_models(self):
        lines = ['可选 GGUF 模型 (HuggingFace, 国内用 hf-mirror.com 镜像):', '']
        for m in HF_MODELS:
            v = ' ✅已验证' if m.get('verified') else ''
            lines.append('• %s  (%s)  %s%s' % (m['file'], m['size'], m['desc'], v))
            lines.append('  下载: https://hf-mirror.com/%s/resolve/main/%s' % (m['repo'], m['file']))
        lines.append('')
        lines.append('下载后点击"选择"按钮加载 .gguf 文件即可启用 Pro 检测。')
        messagebox.showinfo('模型清单', '\n'.join(lines))

    def _bar_color(self, score):
        if score < 30:
            return '#4caf50'
        if score < 55:
            return '#ffc107'
        if score < 75:
            return '#ff9800'
        return '#f44336'

    def detect(self):
        txt = self.text.get('1.0', 'end').strip()
        if not txt:
            messagebox.showwarning('提示', '请先粘贴文本（至少100字符）')
            return
        r = analyze(txt)
        if 'error' in r:
            messagebox.showerror('错误', r['error'])
            return

        # v8: LLM Pro 引擎
        model = self.model_path.get().strip()
        if model:
            self.model_hint.config(text='LLM 推理中... (首次加载较慢)', foreground='#ffc107')
            self.root.update_idletasks()
            llm_res = analyze_llm(txt, model)
            if 'error' in llm_res:
                self.model_hint.config(text='LLM 不可用: %s' % llm_res['error'], foreground='#f44336')
            else:
                r['llm'] = llm_res
                r['ai_probability'] = round(0.6 * r['ai_probability'] + 0.4 * llm_res['llm_score'], 1)
                r['level'] = level_of(r['ai_probability'])
                self.model_hint.config(
                    text='LLM深度: 困惑度 %.1f · 可预测率 %.1f%% · 排名 %.1f%% (分 %.1f)'
                         % (llm_res['ppl'], llm_res['pred_rate'] * 100, llm_res['rank_pct'] * 100, llm_res['llm_score']),
                    foreground='#00e5ff')

        p = r['ai_probability']
        color = LEVEL_COLORS.get(r['level'], '#888888')
        self.score_label.config(text='%.1f%%' % p, foreground=color)
        self.level_label.config(text='%s  %s' % (r['level'], LEVEL_TEXT.get(r['level'], '')),
                                foreground=color)

        flags = '；'.join(r['flags']) if r['flags'] else '未发现明显AI特征'
        self.flag_label.config(text=flags)

        s = r['stats']
        self.stats_label.config(
            text='%d字符 · %d词 · %d句 · %d段 | 语气词%d · 套话%d · 脚手架%d/5 | 语言: %s'
                 % (s['chars'], s['words'], s['sentences'], s['paragraphs'],
                    s['tone_words'], s['templates'], s['scaffold_stages'],
                    '中文' if r['language'] == 'zh' else 'English'))

        for row in self.detail_rows:
            row.destroy()
        self.detail_rows = []

        for d in r['details']:
            row = ttk.Frame(self.detail_frame)
            row.pack(fill='x', pady=1)
            ttk.Label(row, text='%-8s' % d['name'], width=10,
                      foreground='#00e5ff', font=('Sans', 9)).pack(side='left')
            bar = ttk.Progressbar(row, maximum=100, value=d['ai_score'], length=280)
            bar.pack(side='left', padx=6, fill='x', expand=True)
            ttk.Label(row, text='%.1f' % d['ai_score'], width=6,
                      foreground=self._bar_color(d['ai_score']),
                      font=('Sans', 9, 'bold')).pack(side='left')
            self.detail_rows.append(row)


def main():
    root = tk.Tk()
    AigcDetectorApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()