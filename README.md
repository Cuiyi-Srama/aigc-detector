# AIGC 文本检测器 v8 (AIGC Text Detector)

> 纯本地 · 离线运行 · 17 维统计特征 + 6 大模型指纹 + 论证脚手架检测 + 🧠 LLM Pro 深度引擎 + 📑 分文章类型指纹 (v8.2)

一个基于统计特征的 AI 生成内容（AIGC）检测工具。无需联网、无需上传文本，所有检测都在本地完成，保护隐私的同时快速判断一段文本是"人类写作"还是"AI 生成"。

---

## 📑 分文章类型指纹检测 (v8.2 新增)

不同类型的文章，AI 生成风格差异巨大——学术论文本身高度结构化，文学创作则自由奔放。用**全局固定阈值**检测会导致误判：人类论文被判 AI，AI 散文被判人类。

v8.2 引入**类型专属指纹**：检测前选择文章类型，系统使用该类型的人类/AI 语料统计基准（均值±标准差）进行归一化判断：

| 类型 | 说明 | 指纹特性 |
|------|------|---------|
| 🌐 通用 | 不指定，全局混合基准 | 默认 |
| 📚 学术论文/研究报告 | 摘要-引言-方法-结论 | 人类论文也结构化，阈值放宽 |
| 📰 新闻报道/时评 | 客观叙事 | AI 模板化（导语-背景-展望）特征明显 |
| 📖 文学作品/散文 | 文学创作 | 人类困惑度极高波动大，差异最明显 |
| 💬 网络文章/科普 | 自媒体/博客/问答 | 差异中等 |
| 🏛 公文/演讲稿 | 正式文书 | 人类与 AI 都程式化，差异最小，阈值最严 |

**指纹来源**（全部版权安全）：
- **人类语料**：HC3 开源数据集人类回答（金融/医学/法律/百科/开放域）、鲁迅公版散文与杂文（维基文库）、Project Gutenberg 公版英文文学、arXiv 开放论文、美国建国文献
- **AI 语料**：现代主流大模型风格生成（2026 年代表现），覆盖中英双语、5 大类型

**判定逻辑**：每个维度（困惑度/top1/top5 可预测率/分段波动/罕见词率）与所选类型的 `人类均值±σ` 和 `AI均值±σ` 做方向性 sigmoid 归一化 → 类型感知 LLM 深度分 → 与快速检测按 0.45:0.55 融合。

深度检测完成后，报告**直接覆盖**快速检测结果区，显示融合结论与类型基准说明。

---

## 🧠 LLM Pro 深度检测 (v8 新增, 可选)

v8 引入 **LLM Pro 引擎**：加载本地 GGUF 模型（HuggingFace 开源模型），实现 GPTZero 核心原理的真深度检测：

| 指标 | 原理 | AI 文本特征 | 人类文本特征 |
|------|------|------------|------------|
| **困惑度 Perplexity** | 文本对语言模型的"意料程度" | 低（AI 文本在模型分布"低谷"） | 高（人类写作跳跃、意外） |
| **top-1 可预测率** | 模型猜中下一个 token 的比例 | 高（>35%，用词可预测） | 低（<20%） |
| **目标 token 排名** | 真实 token 在候选词中的百分位 | 靠前（<10%，意料之中） | 分散（>25%） |
| **分段困惑度波动** (v8.1) | 各段困惑度的变异系数 | 低（<0.15，风格均匀平滑） | 高（>0.35，段落起伏大） |
| **top-5 可预测率** (v8.1) | 真实 token 进入模型前 5 候选的比例 | 高（>60%） | 低（<40%） |
| **罕见词率** (v8.1) | 模型对真实 token 感到"惊讶"的比例 | 低（<8%，用词保守） | 高（>18%，用词自由） |

检测时 LLM 深度分（top1/top5/排名综合）与 17 维启发式按 **0.45 : 0.55** 融合。

### 安装与使用 (Python 桌面版)

```bash
pip install llama-cpp-python        # 安装推理引擎
python3 aigc_detector.py --list-models                # 查看可选模型
python3 aigc_detector.py -f file.txt --model 模型.gguf  # 启用深度检测
python3 aigc_detector.py --download-model lmstudio-community/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q4_K_M.gguf  # 一键下载
```

> 国内用户默认走 `hf-mirror.com` 镜像直连；可用 `--hf-base https://huggingface.co` 切换官方源。
> 未安装 llama-cpp-python 或未指定模型时，自动降级为 17 维启发式（原有功能不受影响）。

### 可选模型 (HuggingFace)

| 模型 | 大小 | 说明 | 验证 |
|------|------|------|------|
| [gemma-4-E4B-it-Q4_K_M.gguf](https://hf-mirror.com/lmstudio-community/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-Q4_K_M.gguf) | 5.3GB | Gemma4 4B 高效版 (推荐, 中英均衡) | ✅ |
| [gemma-4-E4B-it-Q6_K.gguf](https://hf-mirror.com/lmstudio-community/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-Q6_K.gguf) | 6.2GB | Gemma4 4B 更高精度 | ✅ |
| [gemma-4-E4B-it-Q8_0.gguf](https://hf-mirror.com/lmstudio-community/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-Q8_0.gguf) | 8.0GB | Gemma4 4B 接近无损 | ✅ |
| qwen2.5-1.5b-instruct-q4_k_m.gguf | 1.0GB | 通义千问2.5 1.5B (中文优秀) | |
| qwen2.5-3b-instruct-q4_k_m.gguf | 1.9GB | 通义千问2.5 3B (中文更强) | |
| Llama-3.2-1B-Instruct-Q4_K_M.gguf | 0.9GB | Meta Llama3.2 1B (英文轻量) | |
| Llama-3.2-3B-Instruct-Q4_K_M.gguf | 2.0GB | Meta Llama3.2 3B (英文均衡) | |

模型协议: Gemma4 为 Apache-2.0；下载链接均指向 HuggingFace（含 hf-mirror 镜像）。

## ✨ 特性

- 🧠 **17 维检测维度**：句子突发性、词汇多样性 (TTR)、罕见词比例、模板套话、语气词缺失、主观表达、"的"密度、抽象概念、标点多样性、句式复杂度、段落均匀度、情感表达、数字分节、引号术语、比喻密度、论证脚手架、括号嵌套
- 🔍 **6 大模型指纹库**：DeepSeek / 豆包 / 文心一言 / Kimi / 通义千问 / 腾讯元宝 的风格特征识别
- 📱 **100% 本地离线**：纯前端 JS 实现，文本不出设备，无隐私泄露
- 📊 **可视化仪表盘**：分数仪表盘 + 17 维条形图 + 特征标记 + 统计面板
- 🕐 **检测历史**：自动保存最近 10 次检测记录（localStorage），支持导出
- 🌍 **中英双语支持**：自动识别语言，中文文本启用全维度检测

## 🚀 使用方法

### 方式一：直接打开网页

用浏览器打开 `index.html`（或部署到任意静态服务器 / GitHub Pages）。

### 方式二：Android App

安装 `apk/AIGC-Detector-v6.apk`（WebView 封装，界面与网页版一致）。

### 操作步骤

1. 粘贴文本（**至少 100 字符，越长越准**）
2. 点击「开始检测」
3. 查看 AI 概率分数、判定等级和 17 维明细

> 内置 4 个示例：人类写作（背影）/ 通用 AI / DeepSeek / 人机混合，可一键加载体验。

## 🎯 判定标准

| AI 概率 | 判定 |
|---------|------|
| < 25% | 🟢 Human（人类写作） |
| 25% – 40% | 🟡 Maybe AI（可能含 AI） |
| 40% – 60% | 🟠 Uncertain（不确定） |
| 60% – 80% | 🔴 Likely AI（AI 生成） |
| > 80% | 🔴 Very Likely AI（极可能 AI 生成） |

## 🔬 检测原理

检测器基于 17 个可量化的文本统计特征，加权计算 AI 概率：

| 维度 | 权重 | 原理 |
|------|------|------|
| 句子突发性 Burstiness | 12 | AI 句子长度均匀（CV 低），人类长短句交错（CV 高） |
| 模板套话 | 14 | 检测 "首先/其次/此外/综上所述" 等 AI 高频连接词 |
| 语气词缺失 | 9 | 人类写作常带 "啊/呢/吧/嘛" 等语气词 |
| 罕见词比例 Hapax | 8 | AI 用词保守，罕见词（仅出现一次的词）占比低 |
| 词汇多样性 TTR | 7 | AI 词汇重复率高，多样性低 |
| "的"密度 | 7 | AI 文本 "的" 字使用率异常 |
| 数字分节 | 8 | "一、二、三、" 式机械分节是 AI 标志 |
| 抽象概念 | 6 | "底层逻辑/方法论/赋能" 等抽象词堆砌 |
| 引号术语 | 6 | 大量引号造词（"无中生有""偷换概念"） |
| 主观表达 | 5 | "我觉得/我认为/说实话" 等人类主观标记 |
| 句式复杂度 | 5 | AI 倾向简单均匀句式 |
| 段落均匀度 | 5 | AI 段落长度均匀，人类有起伏 |
| 论证脚手架 | 5 | 破误区→建体系→拆解→练习→升华 五段式 AI 套路 |
| 比喻密度 | 5 | 功能性比喻（"像拼图/像阶梯"）是 AI 高频 |
| 标点多样性 | 4 | 人类标点使用更丰富 |
| 情感表达 | 4 | AI 情感词密度异常（过高或过低） |
| 括号嵌套 | 4 | "（……）" 解释性括号是 AI 高频习惯 |

### 模型指纹

检测器内置 6 个国产大模型的风格特征库（DeepSeek 的"纸老虎/三步走/底层逻辑"、豆包的"总体来说/直白模板"、文心的"随着……的发展/在……背景下"等），命中后叠加模型加成分数。

## ⚠️ 注意事项

- 检测结果基于**简易统计特征**生成，仅供参考，不构成学术判定依据
- 英文文本下部分中文专属维度（语气词/主观表达等）会虚高，英文检测建议以核心维度（Burstiness/TTR/Hapax/模板）为主
- 文本越长检测越准，建议 ≥ 300 字符

## 🖥️ 跨平台版本 (Python CLI)

除网页/APK 外，提供**纯 Python 跨平台版**（Linux / Windows / macOS / Android-Termux 均可运行，零依赖，仅需 Python 3.8+）：

```bash
# Linux / macOS
python3 aigc_detector.py -f file.txt       # 检测文件
python3 aigc_detector.py -t "文本内容"      # 直接检测文本
echo "文本" | python3 aigc_detector.py      # 管道输入
python3 aigc_detector.py -f file.txt --json # JSON 输出 (CI/脚本集成)
python3 aigc_detector.py --demo             # 内置示例演示
python3 aigc_detector.py --quiet            # 精简输出 (仅概率+判定)
python3 aigc_detector.py --dir ./docs       # 批量检测目录 (递归 .txt/.md)

# Windows (PowerShell)
py aigc_detector.py -f file.txt
# 或打包为 exe: pip install pyinstaller && pyinstaller -F aigc_detector.py
```

**v7 跨平台版新增特性：**
- 🇬🇧 英文语气词 / 主观表达 / 情感词库（**修复英文文本虚高误判**）
- 🇬🇧 英文模板套话词库（Moreover / Furthermore / leverage / utilize...）
- 🧩 英文论证脚手架检测（misconception / framework / step-by-step...）
- 📦 `--json` 结构化输出，便于 CI 流水线 / 脚本集成
- 🌐 自动语言识别（中/英），中文全维度检测

## 📁 项目结构

```
aigc-detector/
├── index.html              # Web 版检测器（单文件，含全部逻辑）
├── python/
│   └── aigc_detector.py    # Python 跨平台 CLI 版 (v7)
├── apk/
│   └── AIGC-Detector-v6.apk # Android App 安装包
└── README.md
```

## 📄 License

[MIT](LICENSE)

---

*Made with ❤️ for students & writers who want their words to sound human.*

## 🖼️ GUI 桌面版

提供 tkinter 图形界面（跨平台，零额外依赖）：

```bash
python3 aigc_gui.py          # 直接运行 GUI
```

或使用打包脚本生成免 Python 环境的可执行文件：

- Windows: 双击 `build_windows.bat` → 生成 `dist/AIGC-Detector-v7.exe`
- Linux/macOS: `bash build_linux.sh` → 生成 `dist/AIGC-Detector-v7`

## 🌐 在线版 (GitHub Pages)

无需下载，浏览器直接使用：https://cuiyi-srama.github.io/aigc-detector/
