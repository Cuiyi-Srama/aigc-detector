package com.aigc.detector

/**
 * LLM Pro 引擎 — llama.cpp JNI 封装
 * 加载本地 GGUF, 计算 困惑度/可预测率/排名 (与 Python v8 相同算法)
 */
object LlmProEngine {
    init {
        System.loadLibrary("aigcllm")
    }
    @JvmStatic
    external fun nativeInit(modelPath: String): Boolean
    @JvmStatic
    external fun nativeUnload()
    /** 返回 double[10]: {ppl, pred_rate, rank_pct, llm_score, tokens, seg_std, rare_rate, top5_rate, seg_count, avg_nll}; progress 可为 null */
    @JvmStatic
    external fun nativeAnalyze(text: String, progress: ProgressListener?): DoubleArray

    fun interface ProgressListener {
        fun onProgress(done: Int, total: Int)
    }

    data class Result(
        val ppl: Double,          // [0] 全局困惑度
        val predRate: Double,     // [1] top1 可预测率
        val rankPct: Double,      // [2] 排名中位百分位
        val llmScore: Double,     // [3] LLM 深度分 0-100
        val tokens: Int,          // [4] 评估 token 数
        val segStd: Double,       // [5] 分段困惑度波动 (0-1)
        val rareRate: Double,     // [6] 罕见词率
        val top5Rate: Double,     // [7] top5 可预测率
        val segCount: Int,        // [8] 有效分段数
        val avgNll: Double        // [9] 平均交叉熵
    )
    fun analyze(text: String, progress: ProgressListener? = null): Result? {
        val r = nativeAnalyze(text, progress)
        if (r.size < 10 || r[4] <= 0) return null
        return Result(r[0], r[1], r[2], r[3], r[4].toInt(), r[5], r[6], r[7], r[8].toInt(), r[9])
    }
}