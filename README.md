# AIGC 文本检测器

本地离线运行的 AI 生成内容（AIGC）检测工具。基于文本统计特征分析，可选加载本地大语言模型进行深度检测。所有处理均在本地完成，文本不会上传至任何服务器。

---

## 功能概述

- **统计特征检测**：基于句长分布、词汇多样性、模板套话、标点使用等文本统计特征计算 AI 生成概率
- **深度检测（可选）**：加载本地 GGUF 模型（HuggingFace 开源模型），基于困惑度、可预测率、分段波动等指标检测
- **分类型指纹（v8.2）**：按文章类型（学术 / 新闻 / 文学 / 网络 / 公文）选择对应统计基准，降低跨类型误判
- **中英双语支持**：自动识别语言
- **检测历史**：本地保存最近 10 次检测记录，支持导出

## 分类型指纹（v8.2）

不同类型的文章，AI 生成风格差异较大——学术论文本身高度结构化，文学创作则相对自由。使用统一的全局阈值检测容易产生跨类型误判。

v8.2 引入类型专属指纹：检测前选择文章类型，系统使用该类型的人类 / AI 语料统计基准进行归一化判断。

| 类型 | 说明 |
|------|------|
| 通用 | 不指定类型，使用全局混合基准 |
| 学术论文 / 研究报告 | 摘要-引言-方法-结论结构，专业术语密度高 |
| 新闻报道 / 时评 | 客观叙事与时效性表达 |
| 文学作品 / 散文 | 风格自由，人类写作波动较大 |
| 网络文章 / 科普 | 自媒体 / 博客 / 问答风格 |
| 公文 / 演讲稿 | 正式文书，固定搭配较多 |

**指纹来源**（均为版权安全语料）：

- **人类语料**：HC3 开源数据集人类回答、公版散文与杂文（维基文库）、Project Gutenberg 公版英文文学、arXiv 开放论文、美国建国文献
- **AI 语料**：现代主流大模型生成文本，覆盖中英双语与上述类型

**校准方式**：使用 Qwen3-4B 对 218 篇语料实测特征（困惑度、top-1/top-5 可预测率、分段波动），通过逻辑回归拟合各类型权重（带方向保护与贝叶斯收缩），回测准确率 94.0%。详见 `type_fingerprints.json`。

## 深度检测（LLM Pro）

加载本地 GGUF 模型后，可启用基于语言模型输出的深度检测：

| 指标 | 原理 | AI 文本特征 | 人类文本特征 |
|------|------|------------|------------|
| 困惑度 Perplexity | 文本对语言模型的"意料程度" | 低 | 高 |
| top-1 可预测率 | 模型正确预测下一个 token 的比例 | 高 | 低 |
| 目标 token 排名 | 真实 token 在候选词中的百分位 | 靠前 | 分散 |
| 分段困惑度波动 | 各分段困惑度的变异系数 | 低（风格均匀） | 高（段落起伏大） |
| top-5 可预测率 | 真实 token 进入前 5 候选的比例 | 高 | 低 |
| 罕见词率 | 模型对真实 token 预测困难的比例 | 低 | 高 |

深度检测结果与统计特征检测结果按固定比例融合后输出。

### Python 桌面版安装与使用

```bash
pip install llama-cpp-python        # 安装推理引擎
python3 aigc_detector.py --list-models                # 查看可选模型
python3 aigc_detector.py -f file.txt --model 模型.gguf  # 启用深度检测
python3 aigc_detector.py --download-model lmstudio-community/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q4_K_M.gguf  # 一键下载
```

> 国内用户默认走 `hf-mirror.com` 镜像直连；可用 `--hf-base https://huggingface.co` 切换官方源。
> 未安装 llama-cpp-python 或未指定模型时，自动降级为统计特征检测。

### 可选模型（HuggingFace）

| 模型 | 大小 | 说明 |
|------|------|------|
| [gemma-4-E4B-it-Q4_K_M.gguf](https://hf-mirror.com/lmstudio-community/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-Q4_K_M.gguf) | 5.3GB | Gemma4 4B 高效版（中英均衡） |
| [gemma-4-E4B-it-Q6_K.gguf](https://hf-mirror.com/lmstudio-community/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-Q6_K.gguf) | 6.2GB | Gemma4 4B 更高精度 |
| [gemma-4-E4B-it-Q8_0.gguf](https://hf-mirror.com/lmstudio-community/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-Q8_0.gguf) | 8.0GB | Gemma4 4B 接近无损 |
| qwen2.5-1.5b-instruct-q4_k_m.gguf | 1.0GB | 通义千问 2.5 1.5B（中文） |
| qwen2.5-3b-instruct-q4_k_m.gguf | 1.9GB | 通义千问 2.5 3B（中文） |
| Llama-3.2-1B-Instruct-Q4_K_M.gguf | 0.9GB | Meta Llama 3.2 1B（英文轻量） |
| Llama-3.2-3B-Instruct-Q4_K_M.gguf | 2.0GB | Meta Llama 3.2 3B（英文） |

模型协议：Gemma4 为 Apache-2.0；下载链接指向 HuggingFace（含 hf-mirror 镜像）。

## 使用方法

### 网页版

用浏览器打开 `index.html`（或部署到任意静态服务器 / GitHub Pages）。

### Android App

安装 `apk/` 目录下的 APK（WebView 封装，界面与网页版一致）。

### 操作步骤

1. 粘贴文本（至少 100 字符，越长检测越准确）
2. 点击「开始检测」
3. 查看 AI 概率分数与判定等级

> 内置示例：人类写作 / AI 生成 / 人机混合，可一键加载。

## 判定标准

| AI 概率 | 判定 |
|---------|------|
| < 25% | Human（人类写作） |
| 25% – 40% | Maybe AI（可能含 AI） |
| 40% – 60% | Uncertain（不确定） |
| 60% – 80% | Likely AI（AI 生成） |
| > 80% | Very Likely AI（极可能 AI 生成） |

## 检测原理

检测器基于多项文本统计特征加权计算 AI 概率，主要特征如下：

| 特征 | 原理 |
|------|------|
| 句子突发性 | AI 句子长度均匀，人类长短句交错 |
| 模板套话 | 检测 "首先 / 其次 / 此外 / 综上所述" 等高频连接词 |
| 语气词缺失 | 人类写作常带 "啊 / 呢 / 吧 / 嘛" 等语气词 |
| 罕见词比例 | AI 用词保守，罕见词占比低 |
| 词汇多样性 | AI 词汇重复率较高，多样性低 |
| "的" 密度 | AI 文本 "的" 字使用率异常 |
| 数字分节 | "一、二、三、" 式机械分节常见于 AI 文本 |
| 抽象概念 | "底层逻辑 / 方法论 / 赋能" 等抽象词堆砌 |
| 引号术语 | 大量引号造词（"无中生有" "偷换概念"） |
| 主观表达 | "我觉得 / 我认为 / 说实话" 等人类主观标记 |
| 句式复杂度 | AI 倾向简单均匀句式 |
| 段落均匀度 | AI 段落长度均匀，人类有起伏 |
| 论证脚手架 | 破误区→建体系→拆解→练习→升华 五段式结构 |
| 比喻密度 | 功能性比喻（"像拼图 / 像阶梯"）常见于 AI 文本 |
| 标点多样性 | 人类标点使用更丰富 |
| 情感表达 | AI 情感词密度异常（过高或过低） |
| 括号嵌套 | 解释性括号（"（……）"）常见于 AI 文本 |

### 模型风格特征

检测器内置常见国产大模型（DeepSeek、豆包、文心一言、Kimi、通义千问、腾讯元宝）的高频用语特征库，命中后叠加相应分数。该特征库属于启发式规则，仅作参考。

## 注意事项

- 检测结果基于统计特征生成，仅供参考，不构成学术判定依据
- 英文文本下部分中文专属维度（语气词 / 主观表达等）参考价值有限，英文检测建议以核心维度（句长分布 / 词汇多样性 / 罕见词 / 模板）为主
- 文本越长检测越准，建议不少于 300 字符

## 跨平台命令行版本（Python）

提供纯 Python 跨平台版本（Linux / Windows / macOS / Android-Termux，零依赖，Python 3.8+）：

```bash
# Linux / macOS
python3 aigc_detector.py -f file.txt       # 检测文件
python3 aigc_detector.py -t "文本内容"      # 直接检测文本
echo "文本" | python3 aigc_detector.py      # 管道输入
python3 aigc_detector.py -f file.txt --json # JSON 输出（脚本集成）
python3 aigc_detector.py --demo             # 内置示例
python3 aigc_detector.py --quiet            # 精简输出
python3 aigc_detector.py --dir ./docs       # 批量检测目录

# Windows (PowerShell)
py aigc_detector.py -f file.txt
# 打包 exe: pip install pyinstaller && pyinstaller -F aigc_detector.py
```

GUI 桌面版（tkinter，零额外依赖）：

```bash
python3 aigc_gui.py
```

打包脚本：Windows 用 `build_windows.bat`，Linux/macOS 用 `build_linux.sh`。

## 项目结构

```
aigc-detector/
├── index.html              # Web 版检测器（单文件）
├── android/                # Android App 工程
├── python/
│   ├── aigc_detector.py    # Python CLI 版
│   └── aigc_gui.py         # Python GUI 版
├── apk/                    # 预编译 APK
├── type_fingerprints.json  # 分类型指纹数据（v8.2）
└── README.md
```

## License

[MIT](LICENSE)

## 在线版（GitHub Pages）

https://cuiyi-srama.github.io/aigc-detector/