package com.aigc.detector

import android.app.Activity
import android.app.AlertDialog
import android.content.Intent
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.provider.OpenableColumns
import android.provider.Settings
import android.webkit.JavascriptInterface
import android.webkit.WebView
import android.webkit.WebViewClient
import android.widget.Toast
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.util.ArrayDeque

class MainActivity : Activity() {

    private lateinit var webView: WebView
    private var modelPath: String? = null          // 当前激活模型 (selectModel 切换)
    private var modelPathZh: String? = null        // 中文模型 (千问)
    private var modelPathEn: String? = null        // 英文模型 (Gemma)
    private var curModelLang = "zh"                // 当前语言: zh/en
    private var busy = false
    private var permGuided = false

    // 模型状态机: 0=无模型 1=加载中 2=就绪 3=加载失败
    @Volatile
    private var modelState = 0

    companion object {
        private const val REQ_MODEL = 1001
        private const val PREFS = "aigc_prefs"
        private const val KEY_MODEL = "model_path"
        private const val KEY_MODEL_EN = "model_path_en"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        webView = WebView(this)
        webView.settings.javaScriptEnabled = true
        webView.settings.domStorageEnabled = true
        webView.settings.allowFileAccess = true
        webView.settings.allowContentAccess = true
        webView.webViewClient = object : WebViewClient() {}
        webView.addJavascriptInterface(JsBridge(), "AndroidBridge")
        setContentView(webView)
        webView.loadUrl("file:///android_asset/index.html")

        // 恢复上次选择的双模型并自动预加载 (zh 优先为当前激活)
        val prefs = getSharedPreferences(PREFS, MODE_PRIVATE)
        val savedZh = prefs.getString(KEY_MODEL, null)
        val savedEn = prefs.getString(KEY_MODEL_EN, null)
        if (savedZh != null && setModelPathInternal(savedZh)) {
            modelPathZh = modelPath
        }
        if (savedEn != null && setModelPathEnInternal(savedEn)) {
            // modelPathEn 已由 setModelPathEnInternal 设置
        }
        if (modelPathZh != null || modelPathEn != null) {
            preloadModel()
        }
    }

    override fun onResume() {
        super.onResume()
        // 从权限设置页返回时: 通知 JS 刷新权限状态
        if (::webView.isInitialized) {
            val ok = hasAllFilesAccess()
            runOnUiThread {
                webView.evaluateJavascript("window.__permChanged && window.__permChanged($ok);", null)
            }
        }
        // 首次启动权限引导 (只弹一次)
        if (!permGuided && !hasAllFilesAccess()) {
            permGuided = true
            showPermissionGuide()
        }
    }

    private fun hasAllFilesAccess(): Boolean =
        Build.VERSION.SDK_INT < 30 || Environment.isExternalStorageManager()

    private fun showPermissionGuide() {
        runOnUiThread {
            try {
                AlertDialog.Builder(this)
                    .setTitle("需要「所有文件访问」权限")
                    .setMessage("本应用需要读取手机存储中的 GGUF 大模型文件（如 Models/ 目录下的模型）。\n\n点击「去开启」→ 打开「允许访问所有文件」→ 返回应用即可扫描模型。")
                    .setCancelable(false)
                    .setPositiveButton("去开启") { _, _ -> requestAllFilesAccess() }
                    .setNegativeButton("稍后再说") { _, _ -> }
                    .show()
            } catch (_: Throwable) {}
        }
    }

    /** 引导开启「所有文件访问」权限 (Android 11+) */
    private fun requestAllFilesAccess() {
        try {
            startActivity(Intent(
                Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION,
                Uri.parse("package:$packageName")))
        } catch (_: Throwable) {
            try {
                startActivity(Intent(Settings.ACTION_MANAGE_ALL_FILES_ACCESS_PERMISSION))
            } catch (_: Throwable) {
                toast("请手动到 设置→应用→AIGC 文本检测→权限 中开启「所有文件访问」")
            }
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == REQ_MODEL && resultCode == Activity.RESULT_OK) {
            val uri: Uri? = data?.data
            if (uri != null) {
                val p = resolvePath(uri)
                if (p == null) {
                    toast("无法解析该文件路径，请改用「扫描模型」")
                    return
                }
                if (!setModelPathInternal(p)) {
                    toast("无读取权限，请开启「所有文件访问」后重试")
                    requestAllFilesAccess()
                    return
                }
                notifyModelChosen(File(p).name)
            }
        }
    }

    private fun setModelPathInternal(path: String): Boolean {
        val f = File(path)
        if (!f.exists() || !f.isFile || !path.endsWith(".gguf", ignoreCase = true)) return false
        if (!f.canRead()) return false
        modelPath = f.absolutePath
        getSharedPreferences(PREFS, MODE_PRIVATE).edit().putString(KEY_MODEL, f.absolutePath).apply()
        return true
    }

    /** 设置英文模型路径 (双模型) */
    private fun setModelPathEnInternal(path: String): Boolean {
        val f = File(path)
        if (!f.exists() || !f.isFile || !path.endsWith(".gguf", ignoreCase = true)) return false
        if (!f.canRead()) return false
        modelPathEn = f.absolutePath
        getSharedPreferences(PREFS, MODE_PRIVATE).edit().putString(KEY_MODEL_EN, f.absolutePath).apply()
        return true
    }

    /** 后台预加载模型 (不阻塞 UI, 状态机管理) */
    private fun preloadModel() {
        val p = modelPath ?: return
        if (modelState == 1) return          // 已在加载
        modelState = 1
        Thread {
            try {
                val ok = LlmProEngine.nativeInit(p)
                modelState = if (ok) 2 else 3
                runOnUiThread {
                    webView.evaluateJavascript("window.__llmLoaded && window.__llmLoaded($ok);", null)
                }
            } catch (e: Throwable) {
                modelState = 3
                runOnUiThread {
                    webView.evaluateJavascript("window.__llmLoaded && window.__llmLoaded(false);", null)
                }
            }
        }.start()
    }

    private fun notifyModelChosen(name: String) {
        runOnUiThread {
            webView.evaluateJavascript(
                "window.__llmModelName && window.__llmModelName('${name.replace("'", "\\'")}');", null)
        }
    }

    /** 解析 SAF 的 content:// 为真实路径 (多重兜底) */
    private fun resolvePath(uri: Uri): String? {
        // 1) primary: 格式 (标准 ExternalStorageProvider)
        uri.lastPathSegment?.let { seg ->
            val path = seg.removePrefix("primary:")
            val f = File("/storage/emulated/0/$path")
            if (f.exists() && f.isFile) return f.absolutePath
        }
        // 2) 通过 DISPLAY_NAME 全盘搜索同名文件 (vivo 等私有 provider)
        val name = queryDisplayName(uri) ?: return null
        return findFileByName(name)
    }

    private fun queryDisplayName(uri: Uri): String? = try {
        contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use { c ->
            if (c.moveToFirst()) c.getString(0) else null
        }
    } catch (_: Throwable) { null }

    private fun findFileByName(name: String): String? {
        val stack = ArrayDeque<File>()
        stack.add(File("/storage/emulated/0"))
        var count = 0
        while (stack.isNotEmpty() && count < 300_000) {
            val dir = stack.removeLast()
            val children = dir.listFiles() ?: continue
            for (f in children) {
                count++
                if (f.isDirectory) {
                    if (skipDir(f.name)) continue
                    if (stack.size < 5000) stack.add(f)
                } else if (f.name.equals(name, ignoreCase = true) && f.length() > 50_000_000) {
                    return f.absolutePath
                }
            }
        }
        return null
    }

    private fun skipDir(name: String): Boolean =
        name.startsWith(".") || name == "Android" || name == "obb" || name == "cache" ||
        name == "code_cache" || name == "app_webview" || name == "databases" ||
        name == "shared_prefs" || name == "no_backup"

    private fun toast(msg: String) {
        runOnUiThread { Toast.makeText(this, msg, Toast.LENGTH_SHORT).show() }
    }

    inner class JsBridge {

        /** 是否已授予所有文件访问 */
        @JavascriptInterface
        fun hasAllFilesAccess(): Boolean = this@MainActivity.hasAllFilesAccess()

        /** 跳转权限设置页 */
        @JavascriptInterface
        fun requestAllFilesAccess() { this@MainActivity.requestAllFilesAccess() }

        /** 扫描本地 .gguf 模型 (优先常见目录, 全盘兜底) */
        @JavascriptInterface
        fun listModels(): String {
            val result = JSONArray()
            val seen = HashSet<String>()
            fun scan(dir: File, depth: Int, max: Int) {
                if (result.length() >= max) return
                val children = dir.listFiles() ?: return
                for (f in children) {
                    if (result.length() >= max) return
                    if (f.isDirectory) {
                        if (skipDir(f.name) || depth <= 0) continue
                        if (seen.add(f.absolutePath)) scan(f, depth - 1, max)
                    } else if (f.name.endsWith(".gguf", ignoreCase = true) && f.length() > 50_000_000) {
                        try {
                            val o = JSONObject()
                            o.put("name", f.name)
                            o.put("path", f.absolutePath)
                            o.put("size", f.length() / 1073741824.0)
                            result.put(o)
                        } catch (_: Throwable) {}
                    }
                }
            }
            // 常见模型目录优先 (快)
            for (dir in listOf(
                "/storage/emulated/0/Models",
                "/storage/emulated/0/Download",
                "/storage/emulated/0/下载",
                "/storage/emulated/0/模型")) {
                scan(File(dir), 3, 30)
            }
            // 全盘兜底 (慢)
            if (result.length() == 0) {
                scan(File("/storage/emulated/0"), 7, 30)
            }
            return result.toString()
        }

        /** 设置中文模型路径 (校验存在/可读/.gguf), 持久化 + 后台预加载 */
        @JavascriptInterface
        fun setModelPathZh(path: String): Boolean {
            val ok = setModelPathInternal(path)
            if (ok) {
                modelPathZh = modelPath
                preloadModel()
            }
            return ok
        }

        /** 设置英文模型路径 (校验存在/可读/.gguf), 持久化 + 后台预加载 */
        @JavascriptInterface
        fun setModelPathEn(path: String): Boolean {
            val ok = setModelPathEnInternal(path)
            if (ok) preloadModel()
            return ok
        }

        /** 按语言选择模型: 0=未选 1=加载中 2=就绪 (切换当前激活模型) */
        @JavascriptInterface
        fun selectModel(lang: String): Int {
            curModelLang = lang
            val p = if (lang == "en") modelPathEn else modelPathZh
            if (p == null) return 0
            if (modelState == 1) return 1
            if (modelState == 2 && modelPath == p) return 2
            modelPath = p
            preloadModel()
            return 1
        }

        /** 已选择模型? (任一语言) */
        @JavascriptInterface
        fun hasModel(): Boolean = modelPathZh != null || modelPathEn != null

        /** 当前语言模型名 */
        @JavascriptInterface
        fun modelName(): String = (if (curModelLang == "en") modelPathEn else modelPathZh)?.let { File(it).name } ?: ""

        /** 中文模型名 */
        @JavascriptInterface
        fun modelNameZh(): String = modelPathZh?.let { File(it).name } ?: ""

        /** 英文模型名 */
        @JavascriptInterface
        fun modelNameEn(): String = modelPathEn?.let { File(it).name } ?: ""

        /** 模型状态: 0=无 1=加载中 2=就绪 3=失败 */
        @JavascriptInterface
        fun modelState(): Int = this@MainActivity.modelState

        /** 从系统文件选择器选 (fallback) */
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
            if (busy) {
                notifyJs("", "正在检测中，请稍候...")
                return
            }
            if (modelState == 1) {
                notifyJs("", "模型加载中，请稍候几秒...")
                return
            }
            val p = modelPath ?: run { notifyJs("", "未选择模型"); return }
            busy = true
            Thread {
                try {
                    val ok = LlmProEngine.nativeInit(p)
                    if (!ok) {
                        notifyJs("", "模型加载失败，请确认 GGUF 文件有效")
                        return@Thread
                    }
                    val r = LlmProEngine.analyze(text) { done, total ->
                        runOnUiThread {
                            webView.evaluateJavascript(
                                "window.__llmProgress && window.__llmProgress($done,$total);", null)
                        }
                    }
                    if (r == null) {
                        notifyJs("", "推理失败或文本过短")
                    } else {
                        notifyJs(String.format("%.1f|%.3f|%.3f|%.1f|%d|%.3f|%.3f|%.3f|%d|%.3f",
                            r.ppl, r.predRate, r.rankPct, r.llmScore, r.tokens,
                            r.segStd, r.rareRate, r.top5Rate, r.segCount, r.avgNll), "")
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