// LlmProEngine.cpp — AIGC Detector LLM Pro 引擎 (Android JNI)
// 基于 llama.cpp 加载本地 GGUF, 计算:
//   perplexity(困惑度) / top-1可预测率 / 目标token排名百分位 / LLM深度分
#include <jni.h>
#include <android/log.h>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>

#include "llama.h"

#define TAG "LlmProEngine"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

static llama_model * g_model = nullptr;
static llama_context * g_ctx = nullptr;
static int g_n_vocab = 0;

static void unload() {
    if (g_ctx) { llama_free(g_ctx); g_ctx = nullptr; }
    if (g_model) { llama_model_free(g_model); g_model = nullptr; }
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_aigc_detector_LlmProEngine_nativeInit(JNIEnv * env, jobject, jstring path) {
    unload();
    const char * p = env->GetStringUTFChars(path, nullptr);
    if (!p) return JNI_FALSE;

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;                 // CPU 推理
    g_model = llama_model_load_from_file(p, mp);
    env->ReleaseStringUTFChars(path, p);
    if (!g_model) { LOGE("模型加载失败"); return JNI_FALSE; }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 2048;
    cp.n_batch = 512;
    cp.n_threads = 4;
    cp.n_threads_batch = 4;
    g_ctx = llama_init_from_model(g_model, cp);
    if (!g_ctx) { LOGE("上下文初始化失败"); llama_model_free(g_model); g_model = nullptr; return JNI_FALSE; }

    const llama_vocab * vocab = llama_model_get_vocab(g_model);
    g_n_vocab = vocab ? llama_vocab_n_tokens(vocab) : 0;
    LOGI("模型加载成功, vocab=%d", g_n_vocab);
    return JNI_TRUE;
}

extern "C" JNIEXPORT void JNICALL
Java_com_aigc_detector_LlmProEngine_nativeUnload(JNIEnv *, jobject) {
    unload();
}

// 返回 double[5]: {ppl, pred_rate, rank_pct, llm_score, tokens}
extern "C" JNIEXPORT jdoubleArray JNICALL
Java_com_aigc_detector_LlmProEngine_nativeAnalyze(JNIEnv * env, jobject, jstring text) {
    double zeros[5] = {0, 0, 0, 0, 0};
    jdoubleArray arr = env->NewDoubleArray(5);
    if (!g_model || !g_ctx || g_n_vocab <= 0) { env->SetDoubleArrayRegion(arr, 0, 5, zeros); return arr; }

    const char * t = env->GetStringUTFChars(text, nullptr);
    if (!t) { env->SetDoubleArrayRegion(arr, 0, 5, zeros); return arr; }
    size_t tlen = strlen(t);

    const llama_vocab * vocab = llama_model_get_vocab(g_model);
    std::vector<llama_token> toks(8192);
    int32_t nt = llama_tokenize(vocab, t, (int32_t)tlen, toks.data(), (int32_t)toks.size(), false, false);
    env->ReleaseStringUTFChars(text, t);
    if (nt <= 0) { env->SetDoubleArrayRegion(arr, 0, 5, zeros); return arr; }
    toks.resize(nt);
    if ((int)toks.size() > 1500) toks.resize(1500);

    int n = (int)toks.size() - 1;
    if (n <= 0) { env->SetDoubleArrayRegion(arr, 0, 5, zeros); return arr; }

    llama_batch batch = llama_batch_init(n, 0, 1);
    for (int i = 0; i < n; i++) {
        batch.token[i] = toks[i];
        batch.pos[i]   = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = true;
    }
    batch.n_tokens = n;

    if (llama_decode(g_ctx, batch) != 0) {
        llama_batch_free(batch);
        env->SetDoubleArrayRegion(arr, 0, 5, zeros);
        return arr;
    }
    llama_batch_free(batch);

    const float * logits = llama_get_logits(g_ctx);
    if (!logits) { env->SetDoubleArrayRegion(arr, 0, 5, zeros); return arr; }

    double nll = 0.0;
    double hits = 0.0;
    std::vector<double> ranks;
    ranks.reserve(n);
    const int NV = g_n_vocab;

    for (int i = 0; i < n; i++) {
        const float * lg = logits + (size_t)i * NV;
        int target = toks[i + 1];
        if (target < 0 || target >= NV) continue;

        float best_v = lg[0];
        int best = 0;
        int rank = 0;
        float mx = lg[0];
        for (int v = 1; v < NV; v++) {
            float x = lg[v];
            if (x > best_v) { best_v = x; best = v; }
            if (x > lg[target]) rank++;
            if (x > mx) mx = x;
        }
        if (best == target) hits += 1.0;
        ranks.push_back((double)rank / NV);

        double ex = 0.0;
        for (int v = 0; v < NV; v++) {
            float x = lg[v];
            if (x > mx - 15.0f) ex += exp((double)(x - mx));
        }
        double lse = (double)mx + log(ex);
        nll += lse - (double)lg[target];
    }

    if (ranks.empty()) { env->SetDoubleArrayRegion(arr, 0, 5, zeros); return arr; }

    double ppl = exp(nll / (double)ranks.size());
    double pred_rate = hits / (double)ranks.size();
    std::sort(ranks.begin(), ranks.end());
    double rank_med = ranks[ranks.size() / 2];

    double s_pred = std::min(1.0, pred_rate / 0.35);
    double s_rank = std::min(1.0, std::max(0.0, (0.25 - rank_med) / 0.13));
    double llm_score = std::min(100.0, std::max(0.0, 100.0 * (0.55 * s_pred + 0.45 * s_rank)));

    double out[5] = { ppl, pred_rate, rank_med, llm_score, (double)ranks.size() };
    env->SetDoubleArrayRegion(arr, 0, 5, out);
    LOGI("分析完成: ppl=%.1f pred=%.3f rank=%.3f score=%.1f tokens=%d", ppl, pred_rate, rank_med, llm_score, (int)ranks.size());
    return arr;
}