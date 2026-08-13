// LlmProEngine.cpp — AIGC Detector LLM Pro 引擎 (Android JNI)
// 基于官方 llama.cpp (b10369), 支持 gemma4 等新架构
// v8.1: 多维度输出 + 线程安全 (pthread_mutex)
//   维度: 全局困惑度 / 分段波动 / top1+top5可预测率 / 罕见词惊讶度 / 排名百分位 / LLM深度分
#include <jni.h>
#include <android/log.h>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <pthread.h>
#include <atomic>
#include "llama.h"

#define TAG "LlmProEngine"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

static llama_model * g_model = nullptr;
static llama_context * g_ctx = nullptr;
static int g_n_vocab = 0;
static const int N_BATCH = 512;   // 每批 decode 的 token 数
static const int SEG = 128;       // 分段困惑度的段长 (token)

static pthread_mutex_t g_mutex = PTHREAD_MUTEX_INITIALIZER;
static std::atomic<bool> g_abort{false};   // 看门狗中止标志 (不持锁, 原子安全)

static void unload_locked() {
    if (g_ctx) { llama_free(g_ctx); g_ctx = nullptr; }
    if (g_model) { llama_model_free(g_model); g_model = nullptr; }
    g_n_vocab = 0;
}

static bool is_loaded() { return g_model && g_ctx && g_n_vocab > 0; }

extern "C" JNIEXPORT jboolean JNICALL
Java_com_aigc_detector_LlmProEngine_nativeInit(JNIEnv * env, jobject, jstring path) {
    pthread_mutex_lock(&g_mutex);
    if (is_loaded()) { pthread_mutex_unlock(&g_mutex); return JNI_TRUE; }  // 幂等复用
    unload_locked();
    const char * p = env->GetStringUTFChars(path, nullptr);
    if (!p) { pthread_mutex_unlock(&g_mutex); return JNI_FALSE; }

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;                 // CPU 推理
    g_model = llama_model_load_from_file(p, mp);
    env->ReleaseStringUTFChars(path, p);
    if (!g_model) { LOGE("模型加载失败: %s", p); pthread_mutex_unlock(&g_mutex); return JNI_FALSE; }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 2048;
    cp.n_batch = N_BATCH;
    cp.n_threads = 4;
    cp.n_threads_batch = 4;

    g_ctx = llama_init_from_model(g_model, cp);
    if (!g_ctx) { LOGE("上下文初始化失败"); unload_locked(); pthread_mutex_unlock(&g_mutex); return JNI_FALSE; }

    const llama_vocab * vocab = llama_model_get_vocab(g_model);
    g_n_vocab = vocab ? llama_vocab_n_tokens(vocab) : 0;
    LOGI("模型加载成功, vocab=%d", g_n_vocab);
    pthread_mutex_unlock(&g_mutex);
    return JNI_TRUE;
}

extern "C" JNIEXPORT void JNICALL
Java_com_aigc_detector_LlmProEngine_nativeUnload(JNIEnv *, jobject) {
    pthread_mutex_lock(&g_mutex);
    unload_locked();
    pthread_mutex_unlock(&g_mutex);
}

// 看门狗中止: 设置原子标志, 正在运行的 nativeAnalyze 在下一批循环退出 (不持锁, 可随时调用)
extern "C" JNIEXPORT void JNICALL
Java_com_aigc_detector_LlmProEngine_nativeAbort(JNIEnv *, jobject) {
    g_abort.store(true);
}

// 返回 double[10]:
//   [0] 全局困惑度 ppl
//   [1] top1可预测率 pred_rate
//   [2] 目标token排名中位百分位 rank_pct
//   [3] LLM深度分 llm_score (0-100)
//   [4] 评估token数 tokens
//   [5] 分段困惑度波动 seg_std (归一化 0-1)
//   [6] 罕见词率 rare_rate (target排名>50%的比例)
//   [7] top5可预测率 top5_rate
//   [8] 分段数 seg_count
//   [9] 平均交叉熵 avg_nll
// progress: Java 回调对象 (onProgress(int done,int total)), 可为 null
extern "C" JNIEXPORT jdoubleArray JNICALL
Java_com_aigc_detector_LlmProEngine_nativeAnalyze(JNIEnv * env, jobject, jstring text, jobject progress) {
    double zeros[10] = {0,0,0,0,0,0,0,0,0,0};
    jdoubleArray arr = env->NewDoubleArray(10);
    g_abort.store(false);   // 新一轮分析前重置中止标志

    pthread_mutex_lock(&g_mutex);
    if (!is_loaded()) { pthread_mutex_unlock(&g_mutex); env->SetDoubleArrayRegion(arr, 0, 10, zeros); return arr; }

    jmethodID onProgress = nullptr;
    if (progress != nullptr) {
        jclass cls = env->GetObjectClass(progress);
        onProgress = env->GetMethodID(cls, "onProgress", "(II)V");
        if (onProgress == nullptr) { env->ExceptionClear(); }
    }

    const char * t = env->GetStringUTFChars(text, nullptr);
    if (!t) { pthread_mutex_unlock(&g_mutex); env->SetDoubleArrayRegion(arr, 0, 10, zeros); return arr; }
    size_t tlen = strlen(t);

    const llama_vocab * vocab = llama_model_get_vocab(g_model);
    std::vector<llama_token> toks(8192);
    int32_t nt = llama_tokenize(vocab, t, (int32_t)tlen, toks.data(), (int32_t)toks.size(), false, false);
    env->ReleaseStringUTFChars(text, t);

    if (nt <= 0) { pthread_mutex_unlock(&g_mutex); env->SetDoubleArrayRegion(arr, 0, 10, zeros); return arr; }
    toks.resize(nt);
    if ((int)toks.size() > 1500) toks.resize(1500);
    const int n = (int)toks.size() - 1;
    if (n <= 0) { pthread_mutex_unlock(&g_mutex); env->SetDoubleArrayRegion(arr, 0, 10, zeros); return arr; }

    const int NV = g_n_vocab;
    double nll = 0.0, hits = 0.0, top5_hits = 0.0, rare_cnt = 0.0;
    std::vector<double> ranks;
    ranks.reserve(n);
    // 分段累计: 每 SEG 个预测为一段
    std::vector<double> seg_nll, seg_cnt;
    int n_seg = (n + SEG - 1) / SEG;
    seg_nll.assign(n_seg, 0.0);
    seg_cnt.assign(n_seg, 0.0);

    for (int off = 0; off < n; off += N_BATCH) {
        if (g_abort.load()) {   // 看门狗中止: 放弃剩余批次
            LOGE("分析被看门狗中止 at %d/%d", off, n);
            pthread_mutex_unlock(&g_mutex);
            env->SetDoubleArrayRegion(arr, 0, 10, zeros);
            return arr;
        }
        const int bn = (n - off < N_BATCH) ? (n - off) : N_BATCH;
        llama_batch batch = llama_batch_init(bn, 0, 1);
        for (int i = 0; i < bn; i++) {
            batch.token[i] = toks[off + i];
            batch.pos[i] = off + i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = true;
        }
        batch.n_tokens = bn;

        if (llama_decode(g_ctx, batch) != 0) {
            llama_batch_free(batch);
            LOGE("decode 失败 at %d", off);
            pthread_mutex_unlock(&g_mutex);
            env->SetDoubleArrayRegion(arr, 0, 10, zeros);
            return arr;
        }
        llama_batch_free(batch);

        // 进度回调: 每批完成后通知 Java (锁内回调, 轻量)
        if (onProgress != nullptr) {
            env->CallVoidMethod(progress, onProgress, off + bn, n);
            if (env->ExceptionCheck()) env->ExceptionClear();
        }

        for (int i = 0; i < bn; i++) {
            const int idx = off + i;              // 全局预测序号
            const float * lg = llama_get_logits_ith(g_ctx, i);
            const int target = toks[off + i + 1];
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
            // top5 命中
            int rank5 = 0;
            for (int v = 0; v < NV; v++) if (lg[v] > lg[target]) rank5++;
            if (rank5 < 5) top5_hits += 1.0;
            // 罕见词: target 排名 > 50% (模型对该词很惊讶)
            if ((double)rank / NV > 0.5) rare_cnt += 1.0;

            double rp = (double)rank / NV;
            ranks.push_back(rp);

            double ex = 0.0;
            for (int v = 0; v < NV; v++) {
                float x = lg[v];
                if (x > mx - 15.0f) ex += exp((double)(x - mx));
            }
            double ce = (double)mx + log(ex) - (double)lg[target];
            nll += ce;
            int si = idx / SEG;
            if (si < n_seg) { seg_nll[si] += ce; seg_cnt[si] += 1.0; }
        }
    }

    if (ranks.empty()) { pthread_mutex_unlock(&g_mutex); env->SetDoubleArrayRegion(arr, 0, 10, zeros); return arr; }

    const double total = (double)ranks.size();
    double avg_nll = nll / total;
    double ppl = exp(avg_nll);
    double pred_rate = hits / total;
    double top5_rate = top5_hits / total;
    double rare_rate = rare_cnt / total;

    std::sort(ranks.begin(), ranks.end());
    double rank_med = ranks[ranks.size() / 2];

    // 分段波动: 各段困惑度的变异系数 (std/mean), 0 段数<2
    double seg_std = 0.0;
    int valid_seg = 0;
    {
        std::vector<double> seg_ppl;
        for (int i = 0; i < n_seg; i++) {
            if (seg_cnt[i] >= 16) {
                seg_ppl.push_back(exp(seg_nll[i] / seg_cnt[i]));
                valid_seg++;
            }
        }
        if (valid_seg >= 2) {
            double m = 0.0;
            for (double v : seg_ppl) m += v;
            m /= valid_seg;
            double var = 0.0;
            for (double v : seg_ppl) var += (v - m) * (v - m);
            var /= valid_seg;
            seg_std = (m > 1e-6) ? sqrt(var) / m : 0.0;
            if (seg_std > 1.0) seg_std = 1.0;   // 归一化上限
        }
    }

    // LLM 深度分: top1/top5/rank 综合 (0-100)
    double s_pred = std::min(1.0, pred_rate / 0.35);
    double s_top5 = std::min(1.0, top5_rate / 0.60);
    double s_rank = std::min(1.0, std::max(0.0, (0.25 - rank_med) / 0.13));
    double llm_score = std::min(100.0, std::max(0.0,
        100.0 * (0.40 * s_pred + 0.25 * s_top5 + 0.35 * s_rank)));

    double out[10] = { ppl, pred_rate, rank_med, llm_score, total,
                       seg_std, rare_rate, top5_rate, (double)valid_seg, avg_nll };
    env->SetDoubleArrayRegion(arr, 0, 10, out);
    LOGI("分析完成: ppl=%.1f pred=%.3f top5=%.3f rare=%.3f seg_std=%.3f score=%.1f tokens=%d",
         ppl, pred_rate, top5_rate, rare_rate, seg_std, llm_score, (int)total);
    pthread_mutex_unlock(&g_mutex);
    return arr;
}