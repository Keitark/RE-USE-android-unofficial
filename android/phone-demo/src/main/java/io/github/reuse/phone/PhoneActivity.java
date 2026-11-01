package io.github.reuse.phone;

import android.Manifest;
import android.app.Activity;
import android.content.ClipData;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.graphics.Typeface;
import android.graphics.drawable.GradientDrawable;
import android.media.AudioAttributes;
import android.media.AudioFocusRequest;
import android.media.AudioManager;
import android.media.MediaPlayer;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.view.View;
import android.view.WindowInsets;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.RadioButton;
import android.widget.RadioGroup;
import android.widget.ScrollView;
import android.widget.TextView;
import android.widget.Toast;
import androidx.core.content.FileProvider;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Locale;
import java.util.Map;

/** A bounded phone-microphone comparison. Nothing is recorded until the user taps Start. */
public final class PhoneActivity extends Activity {
    private static final int MIC_PERMISSION = 31;
    private static final int BG = Color.rgb(8, 22, 26);
    private static final int CARD = Color.rgb(18, 36, 41);
    private static final int BORDER = Color.rgb(49, 75, 78);
    private static final int INK = Color.rgb(237, 246, 241);
    private static final int MUTED = Color.rgb(164, 185, 186);
    private static final int MINT = Color.rgb(173, 249, 207);

    private PhoneCaptureController controller;
    private TextView stateTitle, stateDetail, timer, route, inputCount, performance, comparisonDetail, playbackDetail;
    private Button start, stop, before, after, stopPlayback, share;
    private RadioButton twoThreads, fourThreads;
    private ProgressBar level;
    private File latest;
    private MediaPlayer player;
    private AudioManager audioManager;
    private AudioFocusRequest playbackFocus;
    private boolean visible, destroyed, permissionStartPending, active;
    private String phase = "IDLE";

    @Override public void onCreate(Bundle saved) {
        super.onCreate(saved);
        getWindow().setStatusBarColor(BG);
        getWindow().setNavigationBarColor(BG);
        getWindow().getDecorView().setSystemUiVisibility(0);
        if (Build.VERSION.SDK_INT >= 30) getWindow().setDecorFitsSystemWindows(false);
        audioManager = getSystemService(AudioManager.class);
        buildUi();
        controller = new PhoneCaptureController(this, new PhoneCaptureController.Listener() {
            @Override public void onState(PhoneCaptureController.StateSnapshot snapshot) {
                if (destroyed) return;
                active = snapshot.active;
                phase = snapshot.phase;
                showState(snapshot);
                updateControls();
            }
            @Override public void onFinished(File directory, Map<String, Object> statistics) {
                if (destroyed) return;
                active = false;
                phase = "IDLE";
                if (validCapture(directory)) {
                    latest = directory;
                    getPreferences(MODE_PRIVATE).edit().putString("last_capture", directory.getName()).apply();
                    double seconds = number(statistics, "input_samples") / 16000.0;
                    comparisonDetail.setText(String.format(Locale.US,
                            "%.1f seconds saved. Listen to the same recording before and after RE-USE.", seconds));
                    stateTitle.setText("Comparison ready");
                    double rtf = number(statistics, "processing_rtf");
                    long misses = (long) number(statistics, "deadline_misses");
                    long calls = (long) number(statistics, "model_calls");
                    long drops = (long) number(statistics, "dropped_input_samples");
                    stateDetail.setText(calls > 0
                            ? String.format(Locale.US, "Processing used %.0f%% of audio time.\n%d of %d chunks exceeded 100 ms; %d input samples dropped.",
                                    rtf * 100, misses, calls, drops)
                            : "Processed and saved on this phone.");
                    performance.setText(rtf > 1 ? "Saved · processing was slower than incoming audio" : "Saved · average processing kept up");
                    performance.setTextColor(rtf > 1 ? Color.rgb(255, 198, 142) : MINT);
                } else {
                    stateTitle.setText("No audio recorded");
                    stateDetail.setText("Tap Start recording and speak into the phone.");
                }
                level.setProgress(0);
                updateControls();
            }
            @Override public void onError(String message) {
                if (destroyed) return;
                active = false;
                phase = "ERROR";
                stateTitle.setText("Recording stopped");
                stateDetail.setText(message == null ? "Unable to record. Please try again." : message);
                level.setProgress(0);
                updateControls();
            }
        });
        restoreCapture();
        updateControls();
    }

    private void buildUi() {
        ScrollView scroll = new ScrollView(this);
        scroll.setBackgroundColor(BG);
        scroll.setFillViewport(true);
        scroll.setClipToPadding(false);
        scroll.setOnApplyWindowInsetsListener((view, insets) -> {
            int left, top, right, bottom;
            if (Build.VERSION.SDK_INT >= 30) {
                android.graphics.Insets bars = insets.getInsets(WindowInsets.Type.systemBars() | WindowInsets.Type.displayCutout());
                left = bars.left; top = bars.top; right = bars.right; bottom = bars.bottom;
            } else {
                left = insets.getSystemWindowInsetLeft(); top = insets.getSystemWindowInsetTop();
                right = insets.getSystemWindowInsetRight(); bottom = insets.getSystemWindowInsetBottom();
            }
            view.setPadding(left, top, right, bottom);
            return insets;
        });
        LinearLayout content = column();
        content.setPadding(dp(24), dp(24), dp(24), dp(24));
        scroll.addView(content);
        TextView brand = label("RE-USE", 14, MINT, true);
        brand.setLetterSpacing(.18f);
        content.addView(brand);
        TextView title = label("RE-USE Android Library Demo", 30, INK, true);
        margin(content, title, 12, 0);
        margin(content, label("Record your voice. Compare the original with RE-USE enhancement, processed on this phone.", 16, MUTED, false), 12, 22);

        LinearLayout capture = card();
        content.addView(capture);
        stateTitle = label("Ready to record", 22, INK, true);
        capture.addView(stateTitle);
        stateDetail = label("Speak into the phone after recording starts. Each recording lasts up to 60 seconds.", 15, MUTED, false);
        margin(capture, stateDetail, 8, 18);
        timer = label("00:00 / 01:00", 31, MINT, true);
        capture.addView(timer);
        inputCount = label("0 audio samples", 13, MUTED, false);
        margin(capture, inputCount, 5, 8);
        performance = label("Processing speed appears during recording", 13, MUTED, false);
        margin(capture, performance, 0, 14);
        level = new ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal);
        level.setMax(1000);
        level.setProgressTintList(android.content.res.ColorStateList.valueOf(MINT));
        level.setProgressBackgroundTintList(android.content.res.ColorStateList.valueOf(BORDER));
        level.setContentDescription("Microphone input level");
        capture.addView(level, new LinearLayout.LayoutParams(-1, dp(8)));
        route = label("Input: phone microphone", 13, MUTED, false);
        margin(capture, route, 12, 16);
        start = button("Start recording", true);
        start.setContentDescription("Start recording from the phone microphone");
        start.setOnClickListener(v -> requestStart());
        capture.addView(start, fullButton());
        stop = button("Stop & save", false);
        stop.setOnClickListener(v -> {
            if (controller != null) controller.stop();
        });
        margin(capture, stop, 8, 0, fullButton());
        margin(capture, label("Processing", 13, MUTED, true), 20, 5);
        RadioGroup threads = new RadioGroup(this);
        threads.setOrientation(RadioGroup.HORIZONTAL);
        twoThreads = new RadioButton(this);
        fourThreads = new RadioButton(this);
        twoThreads.setId(View.generateViewId()); fourThreads.setId(View.generateViewId());
        twoThreads.setText("2 CPU threads"); fourThreads.setText("4 CPU threads");
        for (RadioButton choice : new RadioButton[]{twoThreads, fourThreads}) {
            choice.setTextColor(INK);
            choice.setTextSize(14);
            choice.setButtonTintList(android.content.res.ColorStateList.valueOf(MINT));
            choice.setMinimumHeight(dp(48));
            threads.addView(choice, new LinearLayout.LayoutParams(0, -2, 1));
        }
        fourThreads.setChecked(true);
        capture.addView(threads);

        LinearLayout comparison = card();
        margin(content, comparison, 18, 0);
        comparison.addView(label("Listen & compare", 22, INK, true));
        comparisonDetail = label("Your before/after recording appears here after you stop.", 15, MUTED, false);
        margin(comparison, comparisonDetail, 9, 16);
        before = button("Play before RE-USE", false);
        before.setOnClickListener(v -> play("raw.wav", "Playing before RE-USE…"));
        comparison.addView(before, fullButton());
        after = button("Play after RE-USE", true);
        after.setOnClickListener(v -> play("enhanced.wav", "Playing after RE-USE…"));
        margin(comparison, after, 9, 0, fullButton());
        playbackDetail = label("", 14, MINT, false);
        margin(comparison, playbackDetail, 10, 0);
        stopPlayback = button("Stop playback", false);
        stopPlayback.setOnClickListener(v -> releasePlayer());
        margin(comparison, stopPlayback, 8, 0, fullButton());
        share = button("Share both WAV files", false);
        share.setOnClickListener(v -> shareRecording());
        margin(comparison, share, 14, 0, fullButton());

        margin(content, label("Recording stops when you leave this screen. Audio stays on this phone. Playback is available after recording, to avoid microphone feedback.", 13, MUTED, false), 20, 0);
        margin(content, label("Unofficial, noncommercial research demo using NVIDIA RE-USE.", 12, MUTED, false), 15, 0);
        setContentView(scroll);
        scroll.requestApplyInsets();
    }

    private void showState(PhoneCaptureController.StateSnapshot s) {
        switch (s.phase) {
            case "PREPARING":
                stateTitle.setText("Getting RE-USE ready…");
                break;
            case "RECORDING":
                stateTitle.setText(s.processingBehind ? "Recording · catching up" : "Recording · processing live");
                break;
            case "STOPPING":
                stateTitle.setText("Saving comparison…");
                break;
            case "ERROR":
                stateTitle.setText("Recording stopped");
                break;
            default:
                stateTitle.setText(validCapture(latest) ? "Comparison ready" : "Ready to record");
                break;
        }
        if (s.message != null && !s.message.isEmpty()) stateDetail.setText(s.message);
        int seconds = (int) Math.max(0, Math.min(60, s.seconds()));
        String clock = String.format(Locale.US, "%02d:%02d / 01:00", seconds / 60, seconds % 60);
        if (!clock.contentEquals(timer.getText())) timer.setText(clock);
        String count = String.format(Locale.US, "%,d audio samples", s.capturedSamples);
        if (!count.contentEquals(inputCount.getText())) inputCount.setText(count);
        String speed;
        if ("RECORDING".equals(s.phase) && s.modelCalls > 0) {
            speed = String.format(Locale.US, "%s · model compute %.0f%% · waiting %.0f ms\n100 ms overruns: %d / %d",
                    s.processingBehind ? "Catching up" : s.modelCalls < 10 ? "Measuring speed" : "Keeping up",
                    s.estimatedProcessingRtf * 100, s.currentQueueMs, s.deadlineMisses, s.modelCalls);
        } else if ("STOPPING".equals(s.phase)) {
            speed = "Finishing accepted audio before saving";
        } else if ("PREPARING".equals(s.phase)) {
            speed = "Preparing · microphone is not recording yet";
        } else {
            speed = "Processing speed appears during recording";
        }
        if (!speed.contentEquals(performance.getText())) performance.setText(speed);
        performance.setTextColor(s.processingBehind ? Color.rgb(255, 198, 142) : MUTED);
        level.setProgress((int) Math.round(Math.min(1, Math.max(0, s.level)) * 1000));
        String input = s.route == null || s.route.isEmpty() ? "Input: phone microphone" : "Input: " + s.route;
        if (!input.contentEquals(route.getText())) route.setText(input);
    }

    private void requestStart() {
        if (controller == null || active || controller.isActive()) return;
        releasePlayer();
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            permissionStartPending = true;
            requestPermissions(new String[]{Manifest.permission.RECORD_AUDIO}, MIC_PERMISSION);
            return;
        }
        beginRecording();
    }

    private void beginRecording() {
        if (!visible || destroyed || controller.isActive()) return;
        active = true;
        phase = "PREPARING";
        stateTitle.setText("Getting RE-USE ready…");
        stateDetail.setText("The microphone opens after the model is ready.");
        performance.setText("Preparing · microphone is not recording yet");
        timer.setText("00:00 / 01:00"); inputCount.setText("0 audio samples"); level.setProgress(0);
        updateControls();
        try {
            controller.start(twoThreads.isChecked() ? 2 : 4);
        } catch (RuntimeException error) {
            active = false; phase = "ERROR";
            stateTitle.setText("Unable to start");
            stateDetail.setText(error.getMessage() == null ? "Please try again." : error.getMessage());
            updateControls();
        }
    }

    @Override public void onRequestPermissionsResult(int request, String[] permissions, int[] results) {
        super.onRequestPermissionsResult(request, permissions, results);
        if (request != MIC_PERMISSION) return;
        boolean wanted = permissionStartPending;
        permissionStartPending = false;
        if (results.length > 0 && results[0] == PackageManager.PERMISSION_GRANTED) {
            if (wanted && visible && !destroyed) beginRecording();
        } else {
            stateTitle.setText("Microphone permission needed");
            stateDetail.setText("Allow microphone access to record. Tap Start recording to try again; if Android no longer asks, enable Microphone in this app’s system settings.");
        }
    }

    private void updateControls() {
        if (start == null) return;
        start.setEnabled(!active);
        stop.setEnabled(active && !"STOPPING".equals(phase));
        twoThreads.setEnabled(!active); fourThreads.setEnabled(!active);
        boolean recorded = validCapture(latest) && !active;
        before.setEnabled(recorded); after.setEnabled(recorded); share.setEnabled(recorded);
        stopPlayback.setVisibility(player == null ? View.GONE : View.VISIBLE);
        playbackDetail.setVisibility(player == null ? View.GONE : View.VISIBLE);
        if (active && visible) getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        else getWindow().clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        for (Button button : new Button[]{start, stop, before, after, share}) button.setAlpha(button.isEnabled() ? 1f : .42f);
    }

    private void play(String name, String description) {
        if (active || controller.isActive() || !validCapture(latest) || !visible) return;
        releasePlayer();
        MediaPlayer candidate = new MediaPlayer();
        player = candidate;
        AudioAttributes attributes = new AudioAttributes.Builder().setUsage(AudioAttributes.USAGE_MEDIA)
                .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH).build();
        candidate.setAudioAttributes(attributes);
        candidate.setOnPreparedListener(ready -> {
            if (destroyed || !visible || player != candidate || active) {
                if (player == candidate) releasePlayer();
                return;
            }
            playbackFocus = new AudioFocusRequest.Builder(AudioManager.AUDIOFOCUS_GAIN_TRANSIENT)
                    .setAudioAttributes(attributes)
                    .setOnAudioFocusChangeListener(change -> {
                        if (change == AudioManager.AUDIOFOCUS_LOSS || change == AudioManager.AUDIOFOCUS_LOSS_TRANSIENT)
                            releasePlayer();
                    }).build();
            if (audioManager.requestAudioFocus(playbackFocus) != AudioManager.AUDIOFOCUS_REQUEST_GRANTED) {
                releasePlayer();
                Toast.makeText(this, "Playback is unavailable while another app has audio focus.", Toast.LENGTH_SHORT).show();
                return;
            }
            ready.start();
            playbackDetail.setText(description);
            updateControls();
        });
        candidate.setOnCompletionListener(done -> { if (player == candidate) releasePlayer(); });
        candidate.setOnErrorListener((failed, what, extra) -> {
            if (player == candidate) releasePlayer();
            Toast.makeText(this, "Unable to play this recording.", Toast.LENGTH_SHORT).show();
            return true;
        });
        try {
            candidate.setDataSource(new File(latest, name).getAbsolutePath());
            playbackDetail.setText("Opening recording…");
            updateControls();
            candidate.prepareAsync();
        } catch (IOException | RuntimeException error) {
            releasePlayer();
            Toast.makeText(this, "Unable to open this recording.", Toast.LENGTH_SHORT).show();
        }
    }

    private void releasePlayer() {
        MediaPlayer previous = player;
        player = null;
        if (previous != null) {
            previous.setOnPreparedListener(null); previous.setOnCompletionListener(null); previous.setOnErrorListener(null);
            try { previous.release(); } catch (RuntimeException ignored) { }
        }
        if (playbackFocus != null && audioManager != null) {
            audioManager.abandonAudioFocusRequest(playbackFocus);
            playbackFocus = null;
        }
        if (playbackDetail != null) playbackDetail.setText("");
        updateControls();
    }

    private void shareRecording() {
        if (active || controller.isActive() || !validCapture(latest)) return;
        releasePlayer();
        try {
            ArrayList<Uri> files = new ArrayList<>();
            for (String name : new String[]{"raw.wav", "enhanced.wav"})
                files.add(FileProvider.getUriForFile(this, getPackageName() + ".files", new File(latest, name)));
            Intent send = new Intent(Intent.ACTION_SEND_MULTIPLE).setType("audio/wav")
                    .putParcelableArrayListExtra(Intent.EXTRA_STREAM, files)
                    .putExtra(Intent.EXTRA_SUBJECT, "RE-USE microphone comparison")
                    .addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
            ClipData clip = ClipData.newRawUri("Before RE-USE", files.get(0));
            clip.addItem(new ClipData.Item(files.get(1))); send.setClipData(clip);
            startActivity(Intent.createChooser(send, "Share before / after WAV files"));
        } catch (RuntimeException error) {
            Toast.makeText(this, "No app is available to share these recordings.", Toast.LENGTH_SHORT).show();
        }
    }

    private void restoreCapture() {
        String name = getPreferences(MODE_PRIVATE).getString("last_capture", null);
        if (name == null || name.contains("/") || name.contains("\\")) return;
        File directory = new File(new File(getFilesDir(), "captures"), name);
        if (validCapture(directory)) {
            latest = directory;
            comparisonDetail.setText("Your last before/after recording is ready to play or share.");
        }
    }

    private boolean validCapture(File directory) {
        if (directory == null) return false;
        try {
            File root = new File(getFilesDir(), "captures").getCanonicalFile();
            if (!root.equals(directory.getCanonicalFile().getParentFile())) return false;
            return new File(directory, "raw.wav").length() > 44 && new File(directory, "enhanced.wav").length() > 44;
        } catch (IOException ignored) { return false; }
    }

    @Override protected void onStart() {
        super.onStart(); visible = true; updateControls();
    }
    @Override protected void onStop() {
        visible = false; permissionStartPending = false;
        releasePlayer();
        if (controller != null && controller.isActive()) controller.stop();
        getWindow().clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        super.onStop();
    }
    @Override protected void onDestroy() {
        destroyed = true;
        releasePlayer();
        if (controller != null) controller.close();
        super.onDestroy();
    }

    private double number(Map<String, Object> map, String name) {
        Object value = map.get(name);
        return value instanceof Number ? ((Number) value).doubleValue() : 0;
    }
    private int dp(int value) { return Math.round(value * getResources().getDisplayMetrics().density); }
    private LinearLayout column() {
        LinearLayout layout = new LinearLayout(this); layout.setOrientation(LinearLayout.VERTICAL); return layout;
    }
    private LinearLayout card() {
        LinearLayout layout = column(); layout.setPadding(dp(20), dp(20), dp(20), dp(20));
        layout.setBackground(shape(CARD, BORDER, 18)); return layout;
    }
    private TextView label(String value, int sp, int color, boolean bold) {
        TextView view = new TextView(this); view.setText(value); view.setTextColor(color); view.setTextSize(sp);
        view.setLineSpacing(dp(3), 1);
        if (bold) view.setTypeface(Typeface.create("sans-serif", Typeface.BOLD));
        return view;
    }
    private Button button(String value, boolean primary) {
        Button view = new Button(this); view.setText(value); view.setAllCaps(false); view.setTextSize(16);
        view.setTextColor(primary ? BG : INK); view.setTypeface(Typeface.DEFAULT_BOLD);
        view.setMinimumHeight(dp(56)); view.setMinHeight(dp(56));
        view.setPadding(dp(12), dp(8), dp(12), dp(8));
        view.setBackground(shape(primary ? MINT : CARD, primary ? MINT : BORDER, 12));
        return view;
    }
    private GradientDrawable shape(int fill, int stroke, int radius) {
        GradientDrawable drawable = new GradientDrawable(); drawable.setColor(fill);
        drawable.setCornerRadius(dp(radius)); drawable.setStroke(dp(1), stroke); return drawable;
    }
    private LinearLayout.LayoutParams fullButton() { return new LinearLayout.LayoutParams(-1, dp(56)); }
    private void margin(LinearLayout parent, View child, int top, int bottom) {
        margin(parent, child, top, bottom, new LinearLayout.LayoutParams(-1, -2));
    }
    private void margin(LinearLayout parent, View child, int top, int bottom, LinearLayout.LayoutParams params) {
        params.topMargin = dp(top); params.bottomMargin = dp(bottom); parent.addView(child, params);
    }
}
