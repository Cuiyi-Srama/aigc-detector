package com.aigc.detector

import android.app.Activity
import android.content.Intent
import android.net.Uri
import android.os.Bundle
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
                    toast("无法读取该文件路径，请选择 Download 目录中的 .gguf")
                    return
                }
                val name = File(modelPath!!).name
                toast("已选择模型: $name")
                runOnUiThread {
                    webView.evaluateJavascript(
                        "window.__llmModelName && window.__llmModelName('$name');", null)
                }
            }
        }
    }

    private fun resolvePath(uri: Uri): String? {
        return try {
            val docId = uri.lastPathSegment?.substringAfterLast(':')
            val f = File("/storage/emulated/0/Download/$docId")
            if (f.exists()) f.absolutePath else null
        } catch (_: Throwable) { null }
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
                    val r = LlmProRunner.analyze(this@MainActivity, p, text)
                    if (r == null) {
                        notifyJs("", "推理失败或未解析到PPL")
                    } else {
                        notifyJs(String.format("%.1f|%.1f|%d", r.ppl, r.llmScore, r.tokens), "")
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