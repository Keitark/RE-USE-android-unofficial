package io.github.reuse.live;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.Service;
import android.content.Intent;
import android.content.pm.ServiceInfo;
import android.os.Binder;
import android.os.Build;
import android.os.IBinder;
import android.os.Process;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.security.MessageDigest;
import java.security.SecureRandom;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import org.json.JSONObject;

/** User-started bounded local processing service. The official Even app owns microphone/Bluetooth access. */
public final class LiveService extends Service {
    static final String START = "io.github.reuse.live.START";
    static final String STOP = "io.github.reuse.live.STOP";
    private static final String CHANNEL = "live-audio";
    private static final int NOTIFICATION = 101;
    private static final String MODEL_SHA = "f950dcfb9321191b2bcb3ab9afa3e100c40e5c64dd1de1d30849be4c1aacce32";
    private final LocalBinder binder = new LocalBinder();
    private final ExecutorService preparation = Executors.newSingleThreadExecutor();
    private volatile LoopbackServer server;
    private volatile boolean starting;
    private volatile boolean destroyed;
    private volatile String error;
    private volatile Map<String, Object> latest = new LinkedHashMap<>();
    private long generation;

    public final class LocalBinder extends Binder {
        LiveService service() { return LiveService.this; }
    }
    @Override public IBinder onBind(Intent intent) { return binder; }

    @Override public int onStartCommand(Intent intent, int flags, int startId) {
        if (intent != null && STOP.equals(intent.getAction())) { stopWork(); stopSelf(); return START_NOT_STICKY; }
        if (intent == null || !START.equals(intent.getAction())) return START_NOT_STICKY;
        showForeground();
        synchronized (this) {
            if (server != null || starting) return START_NOT_STICKY;
            starting = true; error = null;
            long epoch = ++generation;
            preparation.execute(() -> prepare(epoch));
        }
        return START_NOT_STICKY;
    }
    private void showForeground() {
        NotificationManager manager = getSystemService(NotificationManager.class);
        manager.createNotificationChannel(new NotificationChannel(CHANNEL, "G2 live processing", NotificationManager.IMPORTANCE_LOW));
        PendingIntent activity = PendingIntent.getActivity(this, 1, new Intent(this, LiveActivity.class),
                PendingIntent.FLAG_UPDATE_CURRENT | PendingIntent.FLAG_IMMUTABLE);
        PendingIntent stop = PendingIntent.getService(this, 2, new Intent(this, LiveService.class).setAction(STOP),
                PendingIntent.FLAG_UPDATE_CURRENT | PendingIntent.FLAG_IMMUTABLE);
        Notification notification = new Notification.Builder(this, CHANNEL)
                .setSmallIcon(android.R.drawable.ic_media_play).setContentTitle("RE-USE G2 Live")
                .setContentText("Local bridge active. Each capture is limited to 60 seconds.")
                .setContentIntent(activity).setOngoing(true)
                .addAction(new Notification.Action.Builder(null, "Stop", stop).build()).build();
        if (Build.VERSION.SDK_INT >= 29) startForeground(NOTIFICATION, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC);
        else startForeground(NOTIFICATION, notification);
    }
    private void prepare(long epoch) {
        LoopbackServer candidate = null;
        try {
            Path model = getFilesDir().toPath().resolve("model-" + MODEL_SHA);
            Files.createDirectories(model);
            for (String name : new String[]{"model.onnx", "manifest.json", "LICENSE-NVIDIA.txt"}) {
                Path part = model.resolve(name + ".part");
                try (InputStream source = getAssets().open("model/" + name)) {
                    Files.copy(source, part, StandardCopyOption.REPLACE_EXISTING);
                }
                Files.move(part, model.resolve(name), StandardCopyOption.REPLACE_EXISTING);
            }
            String declared = new JSONObject(new String(Files.readAllBytes(model.resolve("manifest.json")),
                    java.nio.charset.StandardCharsets.UTF_8)).getString("model_sha256");
            if (!MODEL_SHA.equals(declared) || !MODEL_SHA.equals(sha256(model.resolve("model.onnx"))))
                throw new IOException("Bundled model hash mismatch");
            Path custom = java.nio.file.Paths.get(getApplicationInfo().nativeLibraryDir, "libreuse_ort_ops.so");
            if (!Files.isRegularFile(custom)) throw new IOException("ARM custom operation library is unavailable");
            Path outputRoot = getFilesDir().toPath().resolve("sessions");
            Files.createDirectories(outputRoot);
            String token = newToken();
            candidate = new LoopbackServer(token, path -> getAssets().open("web/" + path), outputRoot,
                    (output, listener) -> {
                        EnhancementSession session = new EnhancementSession(model, output, custom, listener,
                                () -> Process.setThreadPriority(Process.THREAD_PRIORITY_AUDIO));
                        return new LoopbackServer.Engine() {
                            public void start() { session.start(); }
                            public boolean accept(byte[] pcm) { return session.accept16kS16LE(pcm); }
                            public void finish() { session.finish(); }
                            public void close() { session.close(); }
                        };
                    }, status -> latest = new LinkedHashMap<>(status));
            candidate.start(5000, true);
            synchronized (this) {
                if (destroyed || generation != epoch) { candidate.stop(); return; }
                server = candidate; starting = false;
            }
        } catch (Exception failure) {
            if (candidate != null) candidate.stop();
            synchronized (this) {
                if (generation != epoch) return;
                error = failure.getMessage(); starting = false;
            }
            stopForeground(STOP_FOREGROUND_REMOVE);
            stopSelf();
        }
    }
    synchronized String launchUrl() { return server == null ? null : server.launchUrl(); }
    synchronized String stateText() {
        if (error != null) return "Bridge failed: " + error;
        if (starting) return "Preparing local bridge…";
        if (server == null) return "Stopped. Start the local bridge before opening the Hub page.";
        return "Bridge ready on this phone only.\n" + new JSONObject(server.snapshot()).toString();
    }
    private synchronized void stopWork() {
        generation++; starting = false;
        LoopbackServer current = server; server = null;
        if (current != null) current.stop();
        stopForeground(STOP_FOREGROUND_REMOVE);
    }
    @Override public void onDestroy() {
        destroyed = true; stopWork(); preparation.shutdownNow(); super.onDestroy();
    }
    @Override public void onTaskRemoved(Intent rootIntent) { stopWork(); stopSelf(); }
    @Override public void onTimeout(int startId, int fgsType) { stopWork(); stopSelf(); }
    static String newToken() {
        byte[] bytes = new byte[32]; new SecureRandom().nextBytes(bytes);
        StringBuilder value = new StringBuilder(64);
        for (byte item : bytes) value.append(String.format(java.util.Locale.ROOT, "%02x", item & 255));
        return value.toString();
    }
    private static String sha256(Path path) throws Exception {
        MessageDigest hash = MessageDigest.getInstance("SHA-256");
        try (InputStream stream = Files.newInputStream(path)) {
            byte[] buffer = new byte[65536]; int read;
            while ((read = stream.read(buffer)) >= 0) if (read > 0) hash.update(buffer, 0, read);
        }
        StringBuilder value = new StringBuilder(64);
        for (byte item : hash.digest()) value.append(String.format(java.util.Locale.ROOT, "%02x", item & 255));
        return value.toString();
    }
}
