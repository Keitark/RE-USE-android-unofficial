package io.github.reuse.qnnprobe;

import android.app.Activity;
import android.content.ClipData;
import android.content.Intent;
import android.graphics.Color;
import android.graphics.Typeface;
import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;
import android.widget.Toast;
import androidx.core.content.FileProvider;
import java.io.File;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Locale;
import java.util.UUID;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;
import org.json.JSONArray;
import org.json.JSONObject;

/** User-triggered public-audio comparisons, with explicit partial failures. Never opens a mic. */
public final class ResourceProbeActivity extends Activity {
    private static final int BG = Color.rgb(8, 22, 26), INK = Color.rgb(237, 246, 241);
    private static final int MUTED = Color.rgb(164, 185, 186), MINT = Color.rgb(173, 249, 207);
    private static final ExecutorService WORKER = Executors.newSingleThreadExecutor(work -> new Thread(work, "reuse-resource-probe"));
    private static final AtomicBoolean RUNNING = new AtomicBoolean(), CANCELLED = new AtomicBoolean();
    private static volatile String progress = "";
    private static volatile ResourceComparisonRunner.Result completed;
    private final Handler main = new Handler(Looper.getMainLooper());
    private TextView state, detail, results;
    private Button inventory, compare, cancel, share;
    private boolean visible;
    private ResourceComparisonRunner.Result shown;
    private final Runnable refresh = new Runnable() {
        @Override public void run() { if (visible) { refreshUi(); main.postDelayed(this, 400); } }
    };

    @Override public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().setStatusBarColor(BG); getWindow().setNavigationBarColor(BG);
        ScrollView scroll = new ScrollView(this); scroll.setBackgroundColor(BG); scroll.setFitsSystemWindows(true);
        LinearLayout page = new LinearLayout(this); page.setOrientation(LinearLayout.VERTICAL);
        page.setPadding(dp(24), dp(24), dp(24), dp(24)); scroll.addView(page);
        add(page, label("RE-USE · UNOFFICIAL", 14, MINT, true), 0);
        add(page, label("Compute resources", 30, INK, true), 12);
        add(page, label("Compare CPU threads, direct Adreno GPU access and Qualcomm NPU modes using public test data.", 16, MUTED, false), 12);
        add(page, label("No microphone · no Internet · no recording of your voice", 14, MINT, true), 12);
        state = label("Ready to inspect", 23, INK, true); add(page, state, 24);
        detail = label("Keep this screen visible. The complete comparison takes several minutes. Leaving the app requests a stop after the current native call.", 14, MUTED, false);
        add(page, detail, 10);
        inventory = button("Inspect available resources", () -> start(false)); add(page, inventory, 18);
        compare = button("Compare CPU / Adreno / NPU", () -> start(true)); add(page, compare, 10);
        cancel = button("Stop after current call", () -> { CANCELLED.set(true); progress = "Stopping after the current native call…"; });
        add(page, cancel, 10);
        share = button("Share diagnostics (no audio)", this::shareDiagnostics); add(page, share, 18);
        results = label("", 14, MUTED, false); results.setTextIsSelectable(true); add(page, results, 18);
        add(page, label("Availability is not proof of acceleration. Unsupported paths are reported separately. GPU/NPU event counts do not measure hardware utilization, and these tests do not establish microphone latency or audio quality.", 12, MUTED, false), 22);
        add(page, label("NVIDIA Real-time RE-USE, noncommercial research. Not affiliated with or endorsed by NVIDIA or Qualcomm.", 12, MUTED, false), 16);
        setContentView(scroll);
        shown = completed != null ? completed : restoreLatest();
        if (shown != null) render(shown.diagnostics);
        refreshUi();
        if (savedInstanceState == null && getIntent().getBooleanExtra("run_resource_tests", false)) {
            getIntent().removeExtra("run_resource_tests"); main.post(() -> start(true));
        }
    }

    private void start(boolean runComparison) {
        if (!RUNNING.compareAndSet(false, true)) return;
        CANCELLED.set(false); completed = null; progress = "Collecting compute capabilities…";
        WORKER.execute(() -> {
            try {
                if (runComparison) {
                    completed = ResourceComparisonRunner.run(getApplicationContext(), message -> progress = message, CANCELLED::get);
                } else {
                    File directory = new File(new File(getFilesDir(), "captures"), "resource-inventory-" + UUID.randomUUID());
                    Files.createDirectories(directory.toPath());
                    JSONObject report = new JSONObject(); report.put("source", "compute_resource_inventory");
                    report.put("status", "RUNNING"); report.put("microphone_started", false);
                    Files.write(new File(directory, "resource-runs.json").toPath(), report.toString(2).getBytes(StandardCharsets.UTF_8));
                    report.put("inventory", ResourceInventory.collect(getApplicationContext()));
                    report.put("status", CANCELLED.get() ? "CANCELLED" : "INVENTORY_READY");
                    Files.write(new File(directory, "resource-runs.json").toPath(), report.toString(2).getBytes(StandardCharsets.UTF_8));
                    completed = new ResourceComparisonRunner.Result(directory, report);
                }
            } catch (Exception | LinkageError error) {
                progress = "Resource test error: " + error;
            } finally { RUNNING.set(false); }
        });
        refreshUi();
    }

    private void refreshUi() {
        boolean running = RUNNING.get();
        inventory.setEnabled(!running); compare.setEnabled(!running); cancel.setEnabled(running && !CANCELLED.get());
        share.setEnabled(!running && shown != null && shown.directory != null && new File(shown.directory, "resource-runs.json").isFile());
        if (running) { state.setText(CANCELLED.get() ? "Stopping…" : "Testing resources…"); detail.setText(progress); }
        else if (completed != null && completed != shown) { shown = completed; render(shown.diagnostics); }
        else if (!progress.isEmpty() && progress.startsWith("Resource test error:")) { state.setText("Test incomplete"); detail.setText(progress); }
        if (visible && (running || getIntent().getBooleanExtra("keep_screen_on", false)))
            getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        else getWindow().clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
    }

    private void render(JSONObject report) {
        boolean interrupted = "RUNNING".equals(report.optString("status")) && !RUNNING.get();
        state.setText(interrupted ? "Previous test interrupted" : report.optString("status", "Saved diagnostics").replace('_', ' '));
        detail.setText(interrupted ? "Saved partial diagnostics are available. Tests restart only when you press Compare."
                : "Public-input results only. Compare configurations within the same model; changed precision can change the audio.");
        StringBuilder text = new StringBuilder();
        JSONObject inventoryResult = report.optJSONObject("inventory");
        if (inventoryResult != null) {
            text.append(inventoryResult.optString("manufacturer")).append(' ').append(inventoryResult.optString("model"))
                    .append(" · ").append(inventoryResult.optInt("available_processors")).append(" available CPU processors");
            JSONObject cl = inventoryResult.optJSONObject("opencl");
            if (cl != null) {
                JSONArray platforms = cl.optJSONArray("platforms");
                if (platforms != null) for (int p = 0; p < platforms.length(); p++) {
                    JSONObject platform = platforms.optJSONObject(p);
                    JSONArray devices = platform == null ? null : platform.optJSONArray("devices");
                    if (devices != null) for (int d = 0; d < devices.length(); d++) {
                        JSONObject device = devices.optJSONObject(d); if (device == null) continue;
                        text.append("\nOpenCL GPU: ").append(device.optString("name"))
                                .append(" · reported compute units: ").append(device.opt("compute_units"));
                    }
                }
                if (cl.has("error")) text.append("\nOpenCL: ").append(cl.optString("error"));
            } else if (inventoryResult.has("opencl_error")) text.append("\nOpenCL: ").append(inventoryResult.optString("opencl_error"));
        }
        JSONObject summary = report.optJSONObject("cpu_summary");
        JSONArray configs = summary == null ? null : summary.optJSONArray("configurations");
        if (configs != null) {
            text.append("\n\nCPU configurations (same model per comparison)");
            for (int i = 0; i < configs.length(); i++) {
                JSONObject c = configs.optJSONObject(i); if (c == null) continue;
                String mode = c.optString("cpu_mode");
                text.append("\n").append("cpu_fp32".equals(mode) ? "Original FP32" : "cpu_qdq".equals(mode) ? "Eight-Conv INT8" : mode)
                        .append(" · ").append(c.optInt("threads")).append(" threads: ");
                if (c.optBoolean("eligible", false)) text.append(String.format(Locale.US,
                        "%.2f ms · RTF %.3f\nMax %.2f ms · %d / 130 calls over 100 ms",
                        c.optDouble("mean_call_wall_ms"), c.optDouble("offline_processing_rtf"),
                        c.optDouble("max_call_wall_ms"), c.optInt("calls_over_100ms")));
                else text.append("incomplete or unverified");
            }
        }
        JSONArray htp = report.optJSONArray("htp_trials");
        if (htp != null) for (int i = 0; i < htp.length(); i++) {
            JSONObject t = htp.optJSONObject(i); if (t == null) continue;
            text.append("\n\nNPU ").append(t.optString("htp_performance_mode", t.optString("mode")));
            JSONObject benchmark = t.optJSONObject("benchmark");
            JSONObject backends = benchmark == null ? null : benchmark.optJSONObject("backends");
            if (backends != null) appendTiming(text, backends.optJSONObject("qnn_htp"));
            else text.append("\n").append(t.optString("status", "unavailable"));
        }
        JSONObject gpu = report.optJSONObject("gpu");
        JSONObject vulkan = report.optJSONObject("vulkan");
        if (vulkan != null) {
            text.append("\n\nVulkan compute smoke: ").append(vulkan.optString("status", "see diagnostics"));
            if (vulkan.has("device_name")) text.append("\n").append(vulkan.optString("device_name"));
            text.append("\nA primitive check only; this is not RE-USE acceleration.");
        }
        if (gpu != null) {
            text.append("\n\nDirect GPU: ").append(gpu.optString("status"));
            appendTiming(text, gpu.optJSONObject("full_gpu_hybrid"));
            text.append("\nGPU backend execution verified: ").append(gpu.optBoolean("gpu_backend_execution_verified", false));
        }
        if (report.has("error")) text.append("\n\n").append(report.optString("error"));
        text.append("\n\nCPU time, thermal snapshots, numerical checks and errors are in Share diagnostics.");
        results.setText(text.toString());
    }

    private static void appendTiming(StringBuilder text, JSONObject result) {
        if (result == null) { text.append(" · unavailable"); return; }
        text.append("\n").append(result.optString("status"));
        if (result.has("mean_call_wall_ms")) text.append(String.format(Locale.US, " · %.2f ms", result.optDouble("mean_call_wall_ms")));
        if (result.has("error")) text.append("\n").append(result.optString("error"));
    }

    private ResourceComparisonRunner.Result restoreLatest() {
        File[] directories = new File(getFilesDir(), "captures").listFiles(File::isDirectory);
        if (directories == null) return null;
        java.util.Arrays.sort(directories, (a, b) -> Long.compare(b.lastModified(), a.lastModified()));
        for (File directory : directories) try {
            File file = new File(directory, "resource-runs.json");
            if (file.isFile() && file.length() <= 4 * 1024 * 1024)
                return new ResourceComparisonRunner.Result(directory, new JSONObject(new String(Files.readAllBytes(file.toPath()), StandardCharsets.UTF_8)));
        } catch (Exception ignored) { }
        return null;
    }

    private void shareDiagnostics() {
        if (RUNNING.get() || shown == null || shown.directory == null) return;
        try {
            File root = new File(getFilesDir(), "captures").getCanonicalFile();
            if (!root.equals(shown.directory.getCanonicalFile().getParentFile())) return;
            ArrayList<Uri> files = new ArrayList<>();
            for (String name : new String[]{"resource-runs.json", "stats.json", "phone-capture.json", "qnn-probe.json", "gpu-resource-probe.json",
                    "resource-gpu-isolated-backend.log", "resource-gpu-full-backend.log"}) {
                File file = new File(shown.directory, name);
                if (file.isFile()) files.add(FileProvider.getUriForFile(this, getPackageName() + ".files", file));
            }
            if (files.isEmpty()) return;
            Intent intent = new Intent(Intent.ACTION_SEND_MULTIPLE).setType("application/octet-stream")
                    .putParcelableArrayListExtra(Intent.EXTRA_STREAM, files).addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
            ClipData clip = ClipData.newRawUri("Compute resource diagnostics", files.get(0));
            for (int i = 1; i < files.size(); i++) clip.addItem(new ClipData.Item(files.get(i)));
            intent.setClipData(clip); startActivity(Intent.createChooser(intent, "Share diagnostics without audio"));
        } catch (RuntimeException | java.io.IOException error) { Toast.makeText(this, "Unable to share diagnostics", Toast.LENGTH_LONG).show(); }
    }

    @Override protected void onStart() { super.onStart(); visible = true; main.post(refresh); }
    @Override protected void onStop() {
        visible = false; main.removeCallbacks(refresh); if (RUNNING.get()) CANCELLED.set(true);
        getWindow().clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON); super.onStop();
    }
    private int dp(int value) { return Math.round(value * getResources().getDisplayMetrics().density); }
    private TextView label(String text, int size, int color, boolean bold) {
        TextView v = new TextView(this); v.setText(text); v.setTextColor(color); v.setTextSize(size); v.setLineSpacing(dp(3), 1);
        if (bold) v.setTypeface(Typeface.DEFAULT_BOLD); return v;
    }
    private Button button(String text, Runnable action) {
        Button b = new Button(this); b.setText(text); b.setAllCaps(false); b.setMinHeight(dp(54)); b.setOnClickListener(v -> action.run()); return b;
    }
    private void add(LinearLayout parent, View child, int top) {
        LinearLayout.LayoutParams p = new LinearLayout.LayoutParams(-1, -2); p.topMargin = dp(top); parent.addView(child, p);
    }
}
