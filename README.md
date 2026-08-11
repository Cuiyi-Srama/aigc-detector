# AIGC 文本检测器 v6 (AIGC Text Detector)

> 纯本地 · 离线运行 · 17 维统计特征 + 6 大模型指纹 + 论证脚手架检测

一个基于统计特征的 AI 生成内容（AIGC）检测工具。无需联网、无需上传文本，所有检测都在本地完成，保护隐私的同时快速判断一段文本是"人类写作"还是"AI 生成"。

---

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
