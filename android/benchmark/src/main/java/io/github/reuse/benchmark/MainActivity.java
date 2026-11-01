package io.github.reuse.benchmark;

import android.app.Activity;
import android.os.Bundle;
import android.widget.TextView;

/** The benchmark is driven by instrumentation, with no microphone or network permission. */
public final class MainActivity extends Activity {
    @Override public void onCreate(Bundle state) {
        super.onCreate(state);
        TextView text = new TextView(this);
        text.setPadding(32, 32, 32, 32);
        text.setText("RE-USE Android benchmark\n\n" +
                "Copy an exported model.onnx and manifest.json into:\n" +
                getExternalFilesDir(null) + "/model\n\n" +
                "Run ReuseBenchmarkInstrumentation through adb to measure the full 20 ms " +
                "processing path. Results are written to report.json.\n\n" +
                "This app does not record microphone audio.");
        setContentView(text);
    }
}
