package io.github.reuse.qnnprobe;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.ClipData;
import android.content.ClipboardManager;
import android.content.Intent;
import android.graphics.Color;
import android.graphics.Typeface;
import android.graphics.drawable.GradientDrawable;
import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;
import android.widget.Toast;
import androidx.core.content.FileProvider;
import java.io.File;
import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;
import org.json.JSONArray;
import org.json.JSONObject;

/** An explicit, isolated accelerator test. There is no recording flow in this application. */
public final class QnnProbeActivity extends Activity {
    private static final int BACKGROUND = Color.rgb(8, 22, 26), CARD = Color.rgb(18, 36, 41);
    private static final int INK = Color.rgb(237, 246, 241), MUTED = Color.rgb(164, 185, 186);
    private static final int MINT = Color.rgb(173, 249, 207);
    private static final ExecutorService WORKER = Executors.newSingleThreadExecutor(
            work -> new Thread(work, "reuse-qnn-probe"));
    private static final AtomicBoolean RUNNING = new AtomicBoolean();
    private static volatile String progress = "";
    private static volatile QnnProbeRunner.Result completed;

    private final Handler main = new Handler(Looper.getMainLooper());
    private TextView status, detail, results;
    private Button run, share, copy;
    private QnnProbeRunner.Result displayed;
    private boolean visible;
    private final Runnable refresh = new Runnable() {
        @Override public void run() {
            if (!visible) return;
            refreshUi();
            main.postDelayed(this, 350);
        }
    };

    @Override public void onCreate(Bundle saved) {
        super.onCreate(saved);
        getWindow().setStatusBarColor(BACKGROUND);
        getWindow().setNavigationBarColor(BACKGROUND);
        getWindow().getDecorView().setSystemUiVisibility(0);
        buildUi();
        displayed = completed != null ? completed : restoreLatest();
        if (displayed != null) showResult(displayed);
        refreshUi();
        // Fixed public-fixture benchmark only; never opens a microphone.
        // Used by the package-scoped ADB development runner.
        if (saved == null && getIntent().getBooleanExtra("run_tests", false)) {
            getIntent().removeExtra("run_tests");
            main.post(this::startProbe);
        }
    }

    private void buildUi() {
        ScrollView scroll = new ScrollView(this);
        scroll.setBackgroundColor(BACKGROUND);
        scroll.setFillViewport(true);
        scroll.setFitsSystemWindows(true);
        LinearLayout page = column();
        page.setPadding(dp(24), dp(24), dp(24), dp(24));
        scroll.addView(page);
        add(page, text("RE-USE · UNOFFICIAL", 14, MINT, true), 0);
        add(page, text("Hybrid Streaming Test", 30, INK, true), 12);
        add(page, text("Verify convolution graphs on the NPU, then compare complete RE-USE processing on the public test clip.", 16, MUTED, false), 12);
        add(page, text("Public test audio only · no microphone recording", 14, MINT, true), 12);

        LinearLayout card = column();
        card.setPadding(dp(20), dp(20), dp(20), dp(20));
        GradientDrawable background = new GradientDrawable();
        background.setColor(CARD); background.setCornerRadius(dp(20));
        background.setStroke(dp(1), Color.rgb(49, 75, 78));
        card.setBackground(background);
        add(page, card, 22);
        status = text("Ready to test", 23, INK, true);
        add(card, status, 0);
        detail = text("The isolated prerequisite runs first. The full-model test then uses separate CPU and hybrid sessions with chronological recurrent states.", 15, MUTED, false);
        detail.setTextIsSelectable(true);
        add(card, detail, 10);
        run = button("Run convolution + streaming tests", true);
        run.setOnClickListener(view -> startProbe());
        add(card, run, 18);
        results = text("", 14, MUTED, false);
        results.setTextIsSelectable(true);
        add(card, results, 16);
        share = button("Share diagnostics (no audio)", false);
        share.setOnClickListener(view -> shareDiagnostics());
        add(card, share, 16);
        copy = button("Copy diagnostics / error details", false);
        copy.setOnClickListener(view -> copyDiagnostics());
        add(card, copy, 9);
        Button licenses = button("Licenses & notices", false);
        licenses.setOnClickListener(view -> showLicensePicker());
        add(card, licenses, 9);

        add(page, text("CPU uses two threads. The hybrid model uses QNN for the encoder candidate and CPU for the remaining work. Order and thermal conditions are not controlled.", 13, MUTED, false), 20);
        add(page, text("Full-model timing includes analysis and synthesis. Offline processing speed does not establish microphone latency, continuous playout or audio quality.", 13, MUTED, false), 12);
        add(page, text("Uses NVIDIA model-derived weights for noncommercial research. Not affiliated with or endorsed by NVIDIA or Qualcomm.", 12, MUTED, false), 16);
        setContentView(scroll);
    }

    private void startProbe() {
        if (!RUNNING.compareAndSet(false, true)) return;
        completed = null;
        progress = "Preparing test fixtures…";
        boolean groupOnly = getIntent().getBooleanExtra("group_only", false);
        getIntent().removeExtra("group_only");
        boolean pcmTests = getIntent().getBooleanExtra("run_pcm_tests", false);
        getIntent().removeExtra("run_pcm_tests");
        int repetitions = getIntent().getIntExtra("pcm_repetitions", 1);
        getIntent().removeExtra("pcm_repetitions");
        QnnProbeRunner runner = new QnnProbeRunner(getApplicationContext());
        WORKER.execute(() -> {
            try {
                QnnProbeRunner.Result grouped = runner.run(message -> progress = message);
                QnnProbeRunner.Result result = groupOnly ? grouped : new FullStreamingBenchmark(getApplicationContext()).run(grouped,
                        message -> progress = message);
                if (pcmTests && !groupOnly) {
                    try {
                        JSONObject full = result.diagnostics.optJSONObject("full_stream_benchmark");
                        boolean qnnVerified = full != null && full.optBoolean("hybrid_execution_verified", false);
                        result.diagnostics.put("public_pcm_benchmark", PublicPcmBenchmark.runAll(
                                getApplicationContext(), result.directory, message -> progress = message,
                                qnnVerified, repetitions));
                        byte[] data = (result.diagnostics.toString(2) + "\n").getBytes(StandardCharsets.UTF_8);
                        for (String name : new String[]{"stats.json", "phone-capture.json", "qnn-probe.json"}) {
                            java.nio.file.Path target = result.directory.toPath().resolve(name);
                            java.nio.file.Path temporary = target.resolveSibling(name + ".tmp");
                            Files.write(temporary, data);
                            try { Files.move(temporary, target, java.nio.file.StandardCopyOption.REPLACE_EXISTING,
                                    java.nio.file.StandardCopyOption.ATOMIC_MOVE); }
                            catch (java.nio.file.AtomicMoveNotSupportedException error) {
                                Files.move(temporary, target, java.nio.file.StandardCopyOption.REPLACE_EXISTING);
                            }
                        }
                    } catch (Exception error) {
                        progress = "PCM benchmark error: " + error;
                    }
                }
                completed = result;
            }
            finally { RUNNING.set(false); }
        });
        refreshUi();
    }

    private void refreshUi() {
        boolean running = RUNNING.get();
        run.setEnabled(!running);
        if (running) {
            status.setText("Testing hybrid processing…");
            detail.setText(progress);
            results.setText("No microphone is open. Last-stage diagnostics are saved before native execution.");
            if (visible) getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        } else {
            if (visible && getIntent().getBooleanExtra("keep_screen_on", false))
                getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
            else getWindow().clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
            if (completed != null && completed != displayed) {
                displayed = completed;
                showResult(displayed);
            }
        }
        share.setEnabled(!running && validDirectory(displayed == null ? null : displayed.directory));
        copy.setEnabled(!running && displayed != null);
        run.setAlpha(run.isEnabled() ? 1 : .45f);
        share.setAlpha(share.isEnabled() ? 1 : .45f);
        copy.setAlpha(copy.isEnabled() ? 1 : .45f);
    }

    private void showResult(QnnProbeRunner.Result result) {
        JSONObject report = result.diagnostics;
        boolean interrupted = "RUNNING".equals(report.optString("status"));
        boolean comparison = report.optInt("schema_version", 1) >= 2;
        boolean ready = !interrupted && (comparison
                ? report.optBoolean("comparison_all_verified", false) : report.optBoolean("qnn_htp_test_verified", false));
        boolean anyHtp = report.optBoolean("any_htp_graph_verified", report.optBoolean("qnn_htp_test_verified", false));
        JSONObject full = report.optJSONObject("full_stream_benchmark");
        status.setText(interrupted ? "Previous test interrupted" : ready ? "Comparison ready" : "Partial test results");
        if (interrupted) {
            detail.setText("Stopped during " + report.optString("last_stage", "an unknown stage")
                    + ". Saved diagnostics are available below. Tests restart only when you tap Run.");
        } else if (ready) {
            detail.setText(comparison ? "All CPU / NPU graph comparisons verified. Ratios greater than 1 favor the NPU."
                    : "Previous isolated QNN tests passed. Run the updated test to add CPU and convolution comparisons.");
        } else if (anyHtp) {
            detail.setText("QNN HTP access verified. Some graph comparisons failed or are incomplete; successful results remain available.");
        } else {
            String failure = report.optString("failure_stage");
            detail.setText(failure.isEmpty()
                    ? "NPU execution is unverified. Independent CPU results and error details appear below."
                    : "Stopped at " + failure + ". Share or copy diagnostics for the complete error.");
        }
        StringBuilder lines = new StringBuilder();
        JSONObject device = report.optJSONObject("android_device");
        if (device != null) lines.append(device.optString("manufacturer")).append(' ')
                .append(device.optString("model")).append(" · Android ")
                .append(device.optString("android_release")).append('\n');
        lines.append("ORT ").append(report.optString("ort_version", "not loaded"))
                .append(" · QNN plugin ").append(report.optString("qnn_plugin_package_version", "unknown"))
                .append("\nQNN runtime package ").append(report.optString("qnn_runtime_package_version", "unknown"));
        JSONArray cases = report.optJSONArray("cases");
        if (cases != null) for (int i = 0; i < cases.length(); i++) {
            JSONObject test = cases.optJSONObject(i);
            if (test == null) continue;
            lines.append("\n\n").append(test.optString("title", "Graph test"));
            if (comparison) {
                appendBackend(lines, "CPU FP32", test.optJSONObject("cpu_fp32"));
                appendBackend(lines, "CPU quantized", test.optJSONObject("cpu_qdq"));
                appendBackend(lines, "NPU quantized", test.optJSONObject("qnn_htp"));
                if (test.has("cpu_fp32_over_qnn_speedup")) lines.append(String.format(Locale.US,
                        "\nCPU FP32 / NPU: %.2f×", test.optDouble("cpu_fp32_over_qnn_speedup")));
                if (test.has("cpu_qdq_over_qnn_speedup")) lines.append(String.format(Locale.US,
                        "\nCPU quantized / NPU: %.2f×", test.optDouble("cpu_qdq_over_qnn_speedup")));
            } else {
                appendBackend(lines, "NPU quantized", test);
            }
            if (test.has("error") && !test.has("qnn_htp")) lines.append("\n").append(shortError(test.optString("error")));
        }
        JSONObject qnnInitialization = report.optJSONObject("qnn_initialization");
        if (qnnInitialization != null && qnnInitialization.has("error")) lines.append("\n\nQNN initialization: ")
                .append(shortError(qnnInitialization.optString("error")));
        if (report.has("error")) lines.append("\n\n").append(shortError(report.optString("error")));
        if (report.has("diagnostics_write_error")) lines.append("\n\nCould not save all diagnostic files. Copy diagnostics to retain the error.");
        if (full != null) {
            if (!interrupted) {
                status.setText(full.optBoolean("complete", false) ? "Streaming results ready" : "Streaming results incomplete");
                detail.setText("Public-clip results only. Provider execution, processing time and waveform differences are reported separately.");
            }
            lines.append("\n\nFull streaming model · public clip");
            JSONObject backends = full.optJSONObject("backends");
            if (backends != null) {
                appendFullBackend(lines, "CPU FP32", backends.optJSONObject("cpu_fp32"));
                appendFullBackend(lines, "CPU quantized convolutions", backends.optJSONObject("cpu_qdq"));
                appendFullBackend(lines, "Hybrid QNN + CPU", backends.optJSONObject("qnn_htp"));
            }
            if (full.has("error")) lines.append("\n").append(shortError(full.optString("error")));
            lines.append("\n\nWaveform comparisons and full error details are in Share diagnostics.");
        }
        if (report.has("foreground_timing_qualified")
                && !report.optBoolean("foreground_timing_qualified", false))
            lines.append("\n\nStarted with the screen off or device idle. Timing requires a foreground rerun.");
        JSONObject pcm = report.optJSONObject("public_pcm_benchmark");
        if (pcm != null) {
            lines.append("\n\nPaced 16 kHz PCM pipeline: ").append(pcm.optString("status", "see diagnostics"))
                    .append("\nPublic fixture only; no microphone or speaker playout.");
            JSONObject modes = pcm.optJSONObject("backends");
            if (modes != null) for (String key : new String[]{"cpu_fp32", "cpu_qdq", "qnn_htp"}) {
                JSONObject mode = modes.optJSONObject(key);
                if (mode == null) continue;
                lines.append("\n\n").append(key).append(": ").append(mode.optString("status"));
                if (mode.has("processing_rtf")) lines.append(String.format(Locale.US,
                        "\nRTF %.3f · %d drops · %d calls over 100 ms", mode.optDouble("processing_rtf"),
                        mode.optLong("dropped_input_samples"), mode.optLong("deadline_misses")));
            }
        }
        results.setText(lines.toString());
    }

    private static void appendFullBackend(StringBuilder text, String label, JSONObject backend) {
        text.append("\n\n").append(label).append(": ");
        if (backend == null) { text.append("pending"); return; }
        text.append(backend.optString("status", "pending"));
        if (backend.has("mean_call_wall_ms")) text.append(String.format(Locale.US,
                "\nMean %.2f ms · p95 %.2f ms · max %.2f ms",
                backend.optDouble("mean_call_wall_ms"), backend.optDouble("p95_call_wall_ms"),
                backend.optDouble("max_call_wall_ms")));
        if (backend.has("offline_processing_rtf")) text.append(String.format(Locale.US,
                "\nOffline processing RTF %.3f · calls over 100 ms: %d",
                backend.optDouble("offline_processing_rtf"), backend.optInt("calls_over_100ms")));
        if (backend.has("error")) text.append("\n").append(shortError(backend.optString("error")));
    }

    private static void appendBackend(StringBuilder text, String label, JSONObject backend) {
        text.append("\n\n").append(label).append(": ");
        if (backend == null) { text.append("pending"); return; }
        boolean verified = backend.optBoolean("execution_verified", backend.optBoolean("htp_execution_verified", false));
        if (backend.has("mean_call_wall_ms")) {
            text.append(String.format(Locale.US, "%.3f ms mean · %.3f ms p95",
                    backend.optDouble("mean_call_wall_ms"), backend.optDouble("p95_call_wall_ms")));
            if (!verified) text.append(" (unverified)");
        } else text.append(verified ? "verified" : "unverified");
        if (backend.has("session_create_compile_ms")) text.append(String.format(Locale.US,
                "\nSession / compile %.1f ms", backend.optDouble("session_create_compile_ms")));
        if (backend.has("warmup_first_call_wall_ms")) text.append(String.format(Locale.US,
                " · first warm-up %.3f ms", backend.optDouble("warmup_first_call_wall_ms")));
        if (backend.has("error")) text.append("\n").append(backend.optString("failure_stage"))
                .append(": ").append(shortError(backend.optString("error")));
    }

    private QnnProbeRunner.Result restoreLatest() {
        File root = new File(getFilesDir(), "captures");
        File[] directories = root.listFiles(file -> file.isDirectory()
                && file.getName().matches("[0-9]{8}-[0-9]{6}-[a-f0-9]{8}"));
        if (directories == null) return null;
        CaptureOrder.newestFirst(directories);
        for (File directory : directories) {
            try {
                File json = new File(directory, "stats.json");
                if (!validDirectory(directory) || json.length() > 4 * 1024 * 1024) continue;
                JSONObject report = new JSONObject(new String(Files.readAllBytes(json.toPath()), StandardCharsets.UTF_8));
                return new QnnProbeRunner.Result(directory, report);
            } catch (Exception ignored) { }
        }
        return null;
    }

    private boolean validDirectory(File directory) {
        if (directory == null) return false;
        try {
            return new File(getFilesDir(), "captures").getCanonicalFile().equals(directory.getCanonicalFile().getParentFile())
                    && new File(directory, "stats.json").isFile()
                    && new File(directory, "phone-capture.json").isFile();
        } catch (Exception error) { return false; }
    }

    private void shareDiagnostics() {
        if (RUNNING.get() || displayed == null || !validDirectory(displayed.directory)) return;
        try {
            ArrayList<Uri> files = new ArrayList<>();
            for (String name : new String[]{"stats.json", "phone-capture.json", "qnn-probe.json"}) {
                File file = new File(displayed.directory, name);
                if (file.isFile()) files.add(FileProvider.getUriForFile(this, getPackageName() + ".files", file));
            }
            Intent send = new Intent(Intent.ACTION_SEND_MULTIPLE).setType("application/json")
                    .putParcelableArrayListExtra(Intent.EXTRA_STREAM, files)
                    .putExtra(Intent.EXTRA_SUBJECT, "RE-USE QNN NPU probe diagnostics (no audio)")
                    .addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
            ClipData clip = ClipData.newRawUri("QNN probe diagnostics", files.get(0));
            for (int i = 1; i < files.size(); i++) clip.addItem(new ClipData.Item(files.get(i)));
            send.setClipData(clip);
            startActivity(Intent.createChooser(send, "Share QNN test diagnostics"));
        } catch (RuntimeException error) {
            Toast.makeText(this, "Unable to share. Use Copy diagnostics instead.", Toast.LENGTH_SHORT).show();
        }
    }

    private void copyDiagnostics() {
        if (RUNNING.get() || displayed == null) return;
        ClipboardManager clipboard = getSystemService(ClipboardManager.class);
        if (clipboard != null) {
            clipboard.setPrimaryClip(ClipData.newPlainText("QNN probe diagnostics", displayed.diagnostics.toString()));
            Toast.makeText(this, "Diagnostics and complete error details copied", Toast.LENGTH_SHORT).show();
        }
    }

    private void showLicensePicker() {
        String[] labels = {"Component versions and license guide", "NVIDIA NSCLv1",
                "Qualcomm AI Stack License", "Qualcomm notices", "QNN provider MIT license",
                "QNN provider third-party notices", "ONNX Runtime MIT license",
                "ONNX Runtime third-party notices", "Apache License 2.0"};
        String[] filenames = {"README.md", "NVIDIA-NSCLv1.txt", "Qualcomm-AI-Stack-License.txt",
                "Qualcomm-NOTICE.txt", "QNN-EP-MIT.txt", "QNN-EP-ThirdPartyNotices.txt",
                "ONNXRuntime-MIT.txt", "ONNXRuntime-ThirdPartyNotices.txt", "Apache-2.0.txt"};
        new AlertDialog.Builder(this).setTitle("Licenses & notices")
                .setItems(labels, (dialog, index) -> showLicense(labels[index], filenames[index]))
                .setNegativeButton("Close", null).show();
    }

    private void showLicense(String title, String filename) {
        try {
            String content;
            try (InputStream input = getAssets().open("qnn-notices/" + filename);
                 ByteArrayOutputStream output = new ByteArrayOutputStream()) {
                byte[] block = new byte[8192];
                int read;
                while ((read = input.read(block)) != -1) {
                    if ((long) output.size() + read > 4 * 1024 * 1024)
                        throw new IOException("Notice exceeds the text viewer limit");
                    output.write(block, 0, read);
                }
                content = new String(output.toByteArray(), StandardCharsets.UTF_8);
            }
            ScrollView scroll = new ScrollView(this);
            TextView body = new TextView(this);
            body.setText(content); body.setTextSize(13); body.setTextIsSelectable(true);
            body.setPadding(dp(20), dp(12), dp(20), dp(16));
            scroll.addView(body);
            new AlertDialog.Builder(this).setTitle(title).setView(scroll)
                    .setPositiveButton("Close", null)
                    .setNeutralButton("Copy", (dialog, which) -> {
                        ClipboardManager clipboard = getSystemService(ClipboardManager.class);
                        if (clipboard != null) clipboard.setPrimaryClip(ClipData.newPlainText(title, content));
                    }).show();
        } catch (Exception error) {
            Toast.makeText(this, "Unable to open the packaged notice: " + filename, Toast.LENGTH_LONG).show();
        }
    }

    @Override protected void onStart() {
        super.onStart(); visible = true;
        main.post(refresh);
    }

    @Override protected void onStop() {
        visible = false; main.removeCallbacks(refresh);
        getWindow().clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        super.onStop();
    }

    private static String shortError(String message) {
        return message.length() > 600 ? message.substring(0, 600) + "…" : message;
    }

    private LinearLayout column() {
        LinearLayout layout = new LinearLayout(this);
        layout.setOrientation(LinearLayout.VERTICAL);
        return layout;
    }

    private TextView text(String content, int size, int color, boolean bold) {
        TextView text = new TextView(this);
        text.setText(content); text.setTextColor(color); text.setTextSize(size);
        text.setTypeface(Typeface.DEFAULT, bold ? Typeface.BOLD : Typeface.NORMAL);
        text.setLineSpacing(dp(3), 1);
        return text;
    }

    private Button button(String title, boolean primary) {
        Button button = new Button(this);
        button.setText(title); button.setAllCaps(false); button.setTextSize(15);
        button.setTextColor(primary ? BACKGROUND : INK);
        button.setMinHeight(dp(54));
        GradientDrawable background = new GradientDrawable();
        background.setColor(primary ? MINT : CARD); background.setCornerRadius(dp(14));
        if (!primary) background.setStroke(dp(1), Color.rgb(49, 75, 78));
        button.setBackground(background);
        return button;
    }

    private void add(LinearLayout parent, android.view.View child, int top) {
        LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(-1, -2);
        params.topMargin = dp(top);
        parent.addView(child, params);
    }

    private int dp(int value) { return Math.round(value * getResources().getDisplayMetrics().density); }
}
