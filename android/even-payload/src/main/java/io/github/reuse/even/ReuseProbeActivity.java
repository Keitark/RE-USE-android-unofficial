package io.github.reuse.even;

import android.app.Activity;
import android.app.KeyguardManager;
import android.content.pm.ApplicationInfo;
import android.graphics.Typeface;
import android.os.Build;
import android.os.Bundle;
import android.os.PowerManager;
import android.util.Log;
import android.view.WindowManager;
import android.widget.TextView;
import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import org.json.JSONObject;

/** Explicit debug-only local-file probe; never records a microphone or accesses the network. */
public final class ReuseProbeActivity extends Activity {
    private static final String TAG = "ReuseEvenProbe";
    private boolean holdScreen;

    @Override public void onCreate(Bundle savedState) {
        super.onCreate(savedState);
        if ((getApplicationInfo().flags & ApplicationInfo.FLAG_DEBUGGABLE) == 0) {
            Log.e(TAG, "Probe refused: application is not debuggable");
            finish();
            return;
        }
        holdScreen = getIntent().getBooleanExtra("holdScreen", false);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        TextView text = new TextView(this);
        text.setTypeface(Typeface.MONOSPACE);
        text.setPadding(24, 24, 24, 24);
        text.setText("Running RE-USE local PCM probe...");
        setContentView(text);
        new Thread(() -> runProbe(text), "reuse-even-probe").start();
    }

    private void runProbe(TextView text) {
        File directory = ReuseBridge.modelDirectory(this);
        JSONObject result = new JSONObject();
        try {
            result.put("schema_version", 1).put("device", Build.MANUFACTURER + " " + Build.MODEL)
                    .put("abi", Build.SUPPORTED_ABIS[0]).put("api_level", Build.VERSION.SDK_INT)
                    .put("started_at_epoch_ms", System.currentTimeMillis()).put("hold_screen", holdScreen)
                    .put("interactive_at_start", getSystemService(PowerManager.class).isInteractive())
                    .put("keyguard_locked_at_start", getSystemService(KeyguardManager.class).isKeyguardLocked());
            if (!ReuseBridge.initialize(this)) throw new IOException("Bridge initialization did not configure RE-USE");
            int requestedModelRate = getIntent().getIntExtra("modelRate", ReuseBridge.sampleRate());
            int requestedThreads = getIntent().getIntExtra("threads", ReuseBridge.threads());
            String requestedRuntimeProfile = null;
            if (getIntent().hasExtra("runtimeProfile")) {
                Object value = getIntent().getExtras().get("runtimeProfile");
                if (!(value instanceof String)) throw new IOException("runtimeProfile intent value must be a string");
                requestedRuntimeProfile = (String) value;
            }
            result.put("requested_model_rate", requestedModelRate).put("requested_threads", requestedThreads)
                    .put("requested_runtime_profile", requestedRuntimeProfile == null ? JSONObject.NULL : requestedRuntimeProfile)
                    .put("warm_launch_requested", getIntent().getBooleanExtra("warmRequested", false));
            if (!ReuseBridge.configureForProbe(this, requestedModelRate, requestedThreads, requestedRuntimeProfile)) {
                throw new IOException("Requested probe configuration was rejected");
            }
            result.put("effective_model_rate", ReuseBridge.sampleRate()).put("effective_threads", ReuseBridge.threads())
                    .put("effective_runtime_profile", ReuseBridge.runtimeProfile())
                    .put("session_reconfigured", ReuseBridge.probeReconfigured())
                    .put("session_reused", !ReuseBridge.probeReconfigured());
            int rate = getIntent().getIntExtra("sampleRate", 16000);
            if (rate != 16000) throw new IOException("The G2 LC3 boundary probe requires 16000 Hz PCM");
            File input = new File(directory, "input.pcm16le");
            File output = new File(directory, "enhanced.pcm16le");
            if (!input.isFile() || input.length() == 0 || input.length() % 2 != 0) {
                throw new IOException("Provide nonempty PCM16LE input at files/reuse/input.pcm16le");
            }
            long start = System.nanoTime();
            JSONObject nativeResult = new JSONObject(ReuseBridge.nativeSelfTest(
                    input.getAbsolutePath(), output.getAbsolutePath(), rate));
            result.put("self_test_wall_ms", (System.nanoTime() - start) / 1e6);
            boolean ok = nativeResult.optBoolean("ok", nativeResult.optBoolean("success", false));
            result.put("ok", ok).put("pcm_sample_rate", rate)
                    .put("model_sample_rate", ReuseBridge.sampleRate()).put("self_test", nativeResult)
                    .put("input_bytes", input.length()).put("output_bytes", output.isFile() ? output.length() : 0);
        } catch (Exception | LinkageError error) {
            try { result.put("ok", false).put("error", error.getClass().getSimpleName() + ": " + error.getMessage()); }
            catch (Exception ignored) { }
            Log.e(TAG, "Probe failed", error);
        }
        try {
            result.put("initialization", new JSONObject(ReuseBridge.status()));
            result.put("diagnostics", ReuseBridge.diagnostics());
            result.put("finished_at_epoch_ms", System.currentTimeMillis());
            result.put("interactive_at_finish", getSystemService(PowerManager.class).isInteractive())
                    .put("keyguard_locked_at_finish", getSystemService(KeyguardManager.class).isKeyguardLocked());
            ReuseBridge.writeAtomically(new File(directory, "probe-result.json"),
                    result.toString(2).getBytes(StandardCharsets.UTF_8));
        } catch (Exception | LinkageError error) { Log.e(TAG, "Could not persist complete probe result", error); }
        final String displayed = result.toString();
        Log.i(TAG, displayed);
        runOnUiThread(() -> {
            if (!holdScreen || isFinishing()) {
                getWindow().clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
            }
            if (!isDestroyed()) text.setText(displayed);
        });
    }

    @Override public void finish() {
        getWindow().clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        super.finish();
    }

    @Override protected void onDestroy() {
        getWindow().clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        super.onDestroy();
    }
}
