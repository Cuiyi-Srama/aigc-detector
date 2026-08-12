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
    /** 返回 double[5]: {ppl, pred_rate, rank_pct, llm_score, tokens}; progress 可为 null */
    @JvmStatic
    external fun nativeAnalyze(text: String, progress: ProgressListener?): DoubleArray

    fun interface ProgressListener {
        fun onProgress(done: Int, total: Int)
    }

    data class Result(
        val ppl: Double,
        val predRate: Double,
        val rankPct: Double,
        val llmScore: Double,
        val tokens: Int
    )
    fun analyze(text: String, progress: ProgressListener? = null): Result? {
        val r = nativeAnalyze(text, progress)
        if (r.size < 5 || r[4] <= 0) return null
        return Result(r[0], r[1], r[2], r[3], r[4].toInt())
    }
}