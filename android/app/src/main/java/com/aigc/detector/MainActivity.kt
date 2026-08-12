package com.aigc.detector
import android.app.Activity
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.provider.Settings
import android.webkit.JavascriptInterface
import android.webkit.WebResourceRequest
import android.webkit.WebView
import android.webkit.WebViewClient
import android.widget.Toast
import java.io.File
class MainActivity : Activity() {

    private lateinit var webView: WebView
    private var modelPath: String? = null
    private var busy = false

    companion object {
        private const val REQ_MODEL = 1001
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        webView = WebView(this)
        webView.settings.javaScriptEnabled = true
        webView.settings.domStorageEnabled = true
        webView.settings.allowFileAccess = true
        webView.settings.allowContentAccess = true
        webView.webViewClient = object : WebViewClient() {
            override fun shouldOverrideUrlLoading(view: WebView, request: WebResourceRequest): Boolean {
                return false
            }
        }
        webView.addJavascriptInterface(JsBridge(), "AndroidBridge")
        setContentView(webView)
        webView.loadUrl("file:///android_asset/index.html")
    }

    override fun onDestroy() {
        super.onDestroy()
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == REQ_MODEL && resultCode == Activity.RESULT_OK) {
            val uri: Uri? = data?.data
            if (uri != null) {
                modelPath = resolvePath(uri)
                if (modelPath == null) {
                    toast("无法解析该文件路径")
                    return
                }
                val f = File(modelPath!!)
                if (!f.canRead()) {
                    toast("无读取权限，请开启「所有文件访问」后重试")
                    requestAllFilesAccess()
                    modelPath = null
                    return
                }
                toast("已选择模型: ${f.name} (${f.length() / 1073741824.0}GB)")
                runOnUiThread {
                    webView.evaluateJavascript(
                        "window.__llmModelName && window.__llmModelName('${f.name}');", null)
                }
            }
        }
    }

    /** 解析 SAF 的 content:// 为真实路径 (支持任意主存储目录, 如 /Models/) */
    private fun resolvePath(uri: Uri): String? {
        return try {
            val seg = uri.lastPathSegment ?: return null
            // 格式: primary:Models/gemma-xxx.gguf 或 primary:Download/xxx.gguf
            val path = seg.removePrefix("primary:")
            val f = File("/storage/emulated/0/$path")
            if (f.exists()) f.absolutePath else null
        } catch (_: Throwable) { null }
    }

    /** 引导开启「所有文件访问」权限 (Android 11+) */
    private fun requestAllFilesAccess() {
        try {
            val intent = Intent(
                Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION,
                Uri.parse("package:$packageName"))
            startActivity(intent)
        } catch (_: Throwable) {
            try {
                startActivity(Intent(Settings.ACTION_MANAGE_ALL_FILES_ACCESS_PERMISSION))
            } catch (_: Throwable) { }
        }
    }

    private fun toast(msg: String) {
        runOnUiThread { Toast.makeText(this, msg, Toast.LENGTH_SHORT).show() }
    }

    inner class JsBridge {
        /** 选择 GGUF 模型文件 */
        @JavascriptInterface
        fun pickModel() {
            runOnUiThread {
                val intent = Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
                    addCategory(Intent.CATEGORY_OPENABLE)
                    type = "*/*"
                }
                startActivityForResult(intent, REQ_MODEL)
            }
        }

        /** 已选择模型? */
        @JavascriptInterface
        fun hasModel(): Boolean = modelPath != null

        /** 模型名 */
        @JavascriptInterface
        fun modelName(): String = modelPath?.let { File(it).name } ?: ""

        /** 加载模型 (首次较慢) */
        @JavascriptInterface
        fun loadModel(): Boolean {
            val p = modelPath ?: return false
            return try {
                LlmProEngine.nativeInit(p)
            } catch (e: Throwable) {
                toast("模型加载失败: ${e.message}")
                false
            }
        }

        /** LLM 深度检测 (异步, 回调 window.__llmResult) */
        @JavascriptInterface
        fun analyze(text: String) {
            if (busy) return
            val p = modelPath ?: run { notifyJs("", "未选择模型"); return }
            busy = true
            Thread {
                try {
                    val ok = LlmProEngine.nativeInit(p)
                    if (!ok) {
                        notifyJs("", "模型加载失败，请确认 GGUF 文件有效")
                        return@Thread
                    }
                    val r = LlmProEngine.analyze(text)
                    if (r == null) {
                        notifyJs("", "推理失败或文本过短")
                    } else {
                        notifyJs(String.format("%.1f|%.3f|%.3f|%.1f|%d",
                            r.ppl, r.predRate, r.rankPct, r.llmScore, r.tokens), "")
                    }
                } catch (e: Throwable) {
                    notifyJs("", e.message ?: "未知错误")
                } finally {
                    busy = false
                }
            }.start()
        }

        private fun notifyJs(result: String, error: String) {
            runOnUiThread {
                val js = "window.__llmResult && window.__llmResult(" +
                        "'${result.replace("'", "\\'")}', '${error.replace("'", "\\'")}');"
                webView.evaluateJavascript(js, null)
            }
        }
    }
}