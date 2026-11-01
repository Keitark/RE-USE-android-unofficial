package io.github.reuse.live;

import android.Manifest;
import android.app.Activity;
import android.content.ClipData;
import android.content.ClipboardManager;
import android.content.ComponentName;
import android.content.Context;
import android.content.Intent;
import android.content.ServiceConnection;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.IBinder;
import android.os.Looper;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;
import android.widget.Toast;
import com.google.zxing.BarcodeFormat;
import com.google.zxing.common.BitMatrix;
import com.google.zxing.qrcode.QRCodeWriter;

/** Starts the service visibly, then lets the user load the local page in official Even Hub. */
public final class LiveActivity extends Activity {
    private final Handler handler = new Handler(Looper.getMainLooper());
    private LiveService service;
    private TextView status, url;
    private ImageView qr;
    private String shownUrl;
    private boolean bound;
    private final ServiceConnection connection = new ServiceConnection() {
        public void onServiceConnected(ComponentName name, IBinder binder) {
            service = ((LiveService.LocalBinder) binder).service(); refresh();
        }
        public void onServiceDisconnected(ComponentName name) { service = null; }
    };
    private final Runnable refreshLoop = new Runnable() {
        public void run() { refresh(); handler.postDelayed(this, 500); }
    };
    @Override public void onCreate(Bundle saved) {
        super.onCreate(saved);
        LinearLayout layout = new LinearLayout(this); layout.setOrientation(LinearLayout.VERTICAL);
        int padding = (int) (20 * getResources().getDisplayMetrics().density);
        layout.setPadding(padding, padding, padding, padding);
        TextView title = new TextView(this); title.setText("RE-USE · G2 microphone"); title.setTextSize(24); layout.addView(title);
        TextView explanation = new TextView(this);
        explanation.setText("Start the bridge, then load its URL in Even Hub's development page. The Hub page requests the glasses microphone and saves a raw/enhanced comparison locally. Each capture stops within 60 seconds.\n\nThis page is for a live test; connection status alone does not verify G2 audio.");
        layout.addView(explanation);
        Button start = new Button(this); start.setText("Start local bridge");
        start.setOnClickListener(view -> {
            if (Build.VERSION.SDK_INT >= 33 && checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED)
                requestPermissions(new String[]{Manifest.permission.POST_NOTIFICATIONS}, 10);
            startForegroundService(new Intent(this, LiveService.class).setAction(LiveService.START));
        }); layout.addView(start);
        Button stop = new Button(this); stop.setText("Stop bridge");
        stop.setOnClickListener(view -> startService(new Intent(this, LiveService.class).setAction(LiveService.STOP))); layout.addView(stop);
        status = new TextView(this); status.setTextIsSelectable(true); layout.addView(status);
        url = new TextView(this); url.setTextIsSelectable(true); layout.addView(url);
        Button copy = new Button(this); copy.setText("Copy Hub URL");
        copy.setOnClickListener(view -> {
            String value = service == null ? null : service.launchUrl();
            if (value == null) { Toast.makeText(this, "Start the bridge first", Toast.LENGTH_SHORT).show(); return; }
            getSystemService(ClipboardManager.class).setPrimaryClip(ClipData.newPlainText("Even Hub local URL", value));
            Toast.makeText(this, "URL copied. Open it in Even Hub.", Toast.LENGTH_SHORT).show();
        }); layout.addView(copy);
        qr = new ImageView(this); qr.setAdjustViewBounds(true); qr.setContentDescription("QR code for local Hub page"); layout.addView(qr);
        TextView privacy = new TextView(this); privacy.setText("Audio stays in this companion's private storage. Stopping the bridge aborts an active recording. Stop capture in the Hub page first to drain its tail. The URL token changes whenever the bridge restarts."); layout.addView(privacy);
        ScrollView scroll = new ScrollView(this); scroll.addView(layout); setContentView(scroll);
    }
    @Override protected void onStart() {
        super.onStart(); bound = bindService(new Intent(this, LiveService.class), connection, Context.BIND_AUTO_CREATE);
        handler.post(refreshLoop);
    }
    @Override protected void onStop() {
        handler.removeCallbacks(refreshLoop);
        if (bound) { unbindService(connection); bound = false; service = null; }
        super.onStop();
    }
    private void refresh() {
        if (status == null) return;
        String statusValue = service == null ? "Bridge not connected" : service.stateText();
        if (!statusValue.contentEquals(status.getText())) status.setText(statusValue);
        String value = service == null ? null : service.launchUrl();
        String urlValue = value == null ? "" : value;
        if (!urlValue.contentEquals(url.getText())) url.setText(urlValue);
        if (value == null) { qr.setVisibility(View.GONE); shownUrl = null; }
        else if (!value.equals(shownUrl)) {
            shownUrl = value;
            try {
                BitMatrix matrix = new QRCodeWriter().encode(value, BarcodeFormat.QR_CODE, 640, 640);
                int[] pixels = new int[640 * 640];
                for (int y = 0; y < 640; y++) for (int x = 0; x < 640; x++) pixels[y * 640 + x] = matrix.get(x, y) ? 0xff000000 : 0xffffffff;
                qr.setImageBitmap(Bitmap.createBitmap(pixels, 640, 640, Bitmap.Config.ARGB_8888)); qr.setVisibility(View.VISIBLE);
            } catch (Exception exception) { qr.setVisibility(View.GONE); }
        }
    }
}
