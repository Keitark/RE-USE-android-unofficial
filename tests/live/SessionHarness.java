package io.github.reuse.live;

import java.io.ByteArrayOutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

/** Host integration checks of the real session adapter and native JNI FIR. */
public final class SessionHarness {
    static void require(boolean value, String text) { if (!value) throw new AssertionError(text); }
    static void await(CountDownLatch latch) throws Exception { require(latch.await(10, TimeUnit.SECONDS), "Timed out"); }
    static long number(Map<String,Object> m, String key) { return ((Number)m.get(key)).longValue(); }
    static final class Result implements EnhancementSession.Listener {
        final CountDownLatch ready = new CountDownLatch(1), done = new CountDownLatch(1);
        final ByteArrayOutputStream audio = new ByteArrayOutputStream();
        Map<String,Object> stats; String error; boolean throwAudio;
        public void onReady() { ready.countDown(); }
        public void onAudio(byte[] pcm) {
            if (throwAudio) throw new IllegalStateException("blocked consumer");
            audio.write(pcm, 0, pcm.length);
        }
        public void onFinished(Map<String,Object> s) { stats = s; done.countDown(); }
        public void onError(String code, String message, Map<String,Object> s) { error = code; stats = s; done.countDown(); }
    }
    static final class Identity implements EnhancementSession.Engine {
        float[] pending; boolean first = true, failFlush, failVerification;
        final AtomicInteger closes = new AtomicInteger();
        CountDownLatch entered, release; boolean block; int delayMs;
        public float[] process(float[] input) throws Exception {
            if (delayMs > 0) Thread.sleep(delayMs);
            if (block) { block = false; entered.countDown(); await(release); }
            float[] result = new float[first ? 640 : 800]; int offset = 0;
            if (!first) { System.arraycopy(pending, 0, result, 0, 160); offset = 160; }
            System.arraycopy(input, 0, result, offset, 640);
            pending = Arrays.copyOfRange(input, 640, 800); first = false; return result;
        }
        public float[] flush() throws Exception {
            if (failFlush) throw new IllegalStateException("flush failure");
            return first ? null : pending;
        }
        public void reset() { first = true; pending = null; }
        public void close() { closes.incrementAndGet(); }
        public void verifyAcceleratorExecution() throws Exception {
            if (failVerification) throw new java.io.IOException("NNAPI delegated no kernels during warm-up");
        }
    }
    static byte[] input(int samples) {
        byte[] data = new byte[samples * 2];
        for (int i=0;i<samples;i++) {
            short value = (short)(Math.sin(i*.037)*14000 + (i%311 == 0 ? 11000 : 0));
            data[2*i]=(byte)value; data[2*i+1]=(byte)(value>>8);
        }
        return data;
    }
    static byte[] directFir(byte[] pcm) {
        if (pcm.length == 0) return pcm;
        int n = pcm.length / 2;
        float[] source = new float[n + (n&1) + 94];
        for(int i=0;i<n;i++)source[i]=(short)((pcm[2*i]&255)|(pcm[2*i+1]<<8))/32768f;
        float[] low = new float[source.length/2], high = new float[source.length + 94], temporary = new float[source.length];
        try(FirResampler down = new FirResampler();FirResampler up = new FirResampler()) {
            int count = down.downsample(source,source.length,low);
            int first = up.upsample(low,count,temporary);System.arraycopy(temporary,0,high,0,first);
            float[] tail = new float[94];up.upsample(new float[47],47,tail);System.arraycopy(tail,0,high,first,94);
        }
        byte[] out = new byte[high.length*2];
        for(int i=0;i<high.length;i++) {
            int value=(int)Math.rint(Math.max(-1f,Math.min(32767f/32768f,high[i]))*32768f);
            out[2*i]=(byte)value;out[2*i+1]=(byte)(value>>8);
        }
        return out;
    }
    static void waves(Path directory, Result result, int rawSamples) throws Exception {
        byte[] raw=Files.readAllBytes(directory.resolve("raw.wav")),out=Files.readAllBytes(directory.resolve("enhanced.wav"));
        require(raw.length==44+rawSamples*2,"Raw WAV length");
        require(new String(raw,0,4,java.nio.charset.StandardCharsets.US_ASCII).equals("RIFF"),"WAV header");
        require(Arrays.equals(Arrays.copyOfRange(out,44,out.length),result.audio.toByteArray()),"Recorded/streamed PCM differ");
        require(Files.size(directory.resolve("stats.json"))>2,"Stats missing");
    }
    static void lengths(Path root) throws Exception {
        for(int n:new int[]{0,1,2,1599,1600,1601,3199,3200,3201,6401}) {
            Path dir=root.resolve("length-"+n);Result result=new Result();Identity engine=new Identity();
            AtomicInteger initialized=new AtomicInteger();
            EnhancementSession session=new EnhancementSession(dir,result,initialized::incrementAndGet,()->engine,8000);
            session.start();await(result.ready);byte[] data=input(n);
            int offset=0,packet=0;
            while(offset<data.length) {
                int size=Math.min(data.length-offset,2*(new int[]{1,17,320,799,2,160}[packet++%6]));
                require(session.accept16kS16LE(Arrays.copyOfRange(data,offset,offset+size),offset/2L),"Packet rejected");offset+=size;
            }
            session.finish();session.finish();await(result.done);session.close();
            require(result.error==null,"Unexpected error "+result.error);
            require(engine.closes.get()==1&&initialized.get()==1,"Resource/initializer lifecycle");
            require(number(result.stats,"output_samples")== (n==0?0:n+(n&1)+188),"Final count "+n);
            require(number(result.stats,"flush_calls")== (n==0?0:1),"Flush count");
            require(number(result.stats,"dropped_input_samples")==0,"Dropped input");
            require(Arrays.equals(result.audio.toByteArray(),directFir(data)),"Fragmentation/FIR tail phase differs for "+n);
            waves(dir,result,n);
            require(!session.accept16kS16LE(new byte[2]),"Audio accepted after finish");
        }
    }
    static void blocked(Path root, boolean overflow) throws Exception {
        Result result=new Result();Identity engine=new Identity();
        engine.entered=new CountDownLatch(1);engine.release=new CountDownLatch(1);
        EnhancementSession session=new EnhancementSession(root,result,()->{},()->engine,8000);
        session.start();await(result.ready);engine.block=true;
        require(session.accept16kS16LE(input(1600)),"Initial audio rejected");await(engine.entered);
        require(session.accept16kS16LE(input(8000)),"Queue capacity rejected");
        if(overflow) require(!session.accept16kS16LE(input(1)),"Overflow accepted"); else session.close();
        engine.release.countDown();await(result.done);
        require((overflow?"INPUT_OVERFLOW":"ABORTED").equals(result.error),"Wrong terminal reason");
        require(number(result.stats,"dropped_input_samples")== (overflow?8001:8000),"Dropped count");
        require(engine.closes.get()==1,"Engine leaked");waves(root,result,1600);
    }
    static void emptyStop(Path root) throws Exception {
        for(int i=0;i<20;i++) {
            Path dir=root.resolve("empty-"+i);Result result=new Result();Identity engine=new Identity();
            CountDownLatch create=new CountDownLatch(1);
            EnhancementSession session=new EnhancementSession(dir,result,()->{},()->{await(create);return engine;},8000);
            session.start();session.finish();create.countDown();await(result.done);
            require(result.error==null&&engine.closes.get()==1,"Empty stop failed");waves(dir,result,0);
            for(String name:new String[]{"raw.wav","enhanced.wav","stats.json"}) Files.delete(dir.resolve(name));
            Files.delete(dir); // Proves capture descriptors are closed on Windows too.
        }
    }
    static void failure(Path root,String kind) throws Exception {
        Result result=new Result();Identity engine=new Identity();
        engine.failFlush=kind.equals("flush");result.throwAudio=kind.equals("listener");
        EnhancementSession session=new EnhancementSession(root,result,()->{},()->engine,8000);
        session.start();await(result.ready);
        if(kind.equals("sequence")) {
            require(session.accept16kS16LE(input(1),0),"First sample rejected");
            require(!session.accept16kS16LE(input(1),4),"Discontinuous input accepted");
        } else if(kind.equals("duration")) {
            require(!session.accept16kS16LE(new byte[(120*16000+1)*2]),"Duration cap ignored");
        } else { require(session.accept16kS16LE(input(1)),"Audio rejected");session.finish(); }
        await(result.done);
        String expected=kind.equals("sequence")?"SEQUENCE_GAP":kind.equals("duration")?"DURATION_LIMIT":"PROCESSING_ERROR";
        require(expected.equals(result.error),"Failure was hidden: "+result.error);
        require(engine.closes.get()==1,"Failure leaked model");
        require(Files.size(root.resolve("raw.wav"))>=44&&Files.size(root.resolve("enhanced.wav"))>=44,"Invalid capture on failure");
    }
    static void nativeContracts() throws Exception {
        FirResampler fir=new FirResampler();
        require(fir.downsample(new float[0],0,new float[0])==0,"Empty down");
        require(fir.upsample(new float[0],0,new float[0])==0,"Empty up");
        boolean rejected=false;try{float[] same=new float[4];fir.downsample(same,4,same);}catch(IllegalStateException expected){rejected=true;}
        require(rejected,"Aliased native FIR accepted");
        rejected=false;try{fir.downsample(new float[3],3,new float[2]);}catch(IllegalStateException expected){rejected=true;}
        require(rejected,"Odd FIR count accepted");fir.close();fir.close();
        rejected=false;try{fir.reset();}catch(IllegalStateException expected){rejected=true;}
        require(rejected,"Closed native FIR accepted");
    }
    static void concurrentSnapshots(Path root) throws Exception {
        Result result=new Result();Identity engine=new Identity();engine.delayMs=2;
        EnhancementSession session=new EnhancementSession(root,result,()->{},()->engine,8000);
        session.start();await(result.ready);
        AtomicReference<Throwable> failure=new AtomicReference<>();AtomicInteger snapshots=new AtomicInteger(),liveSnapshots=new AtomicInteger();
        Thread observer=new Thread(()->{
            try {
                while(result.done.getCount()!=0) {
                    Map<String,Object> stats=session.statistics();
                    long queued=number(stats,"queued_input_samples");
                    require(queued>=0&&queued<=8000,"Live queue escaped its bound");
                    require(((Number)stats.get("process_p50_ms")).doubleValue()>=0,"Negative live percentile");
                    require(((Number)stats.get("process_p95_ms")).doubleValue()>=0,"Negative live percentile");
                    snapshots.incrementAndGet();if(number(stats,"model_calls")>0)liveSnapshots.incrementAndGet();
                }
            }catch(Throwable error){failure.set(error);}
        },"session-statistics-observer");
        observer.start();
        int samples=64000;
        for(int offset=0;offset<samples;offset+=1600) {
            while(number(session.statistics(),"queued_input_samples")>6400)Thread.sleep(1);
            require(session.accept16kS16LE(input(1600),offset),"Concurrent snapshot test input rejected");
        }
        session.finish();await(result.done);observer.join(10000);
        require(!observer.isAlive(),"Statistics observer did not stop");
        if(failure.get()!=null)throw new AssertionError("Concurrent statistics failed",failure.get());
        require(snapshots.get()>0&&liveSnapshots.get()>0,"No running-call statistics observed");
        require(result.error==null,"Concurrent observation changed session outcome");
        require(number(result.stats,"input_samples")==samples&&number(result.stats,"processed_input_samples")==samples,"Final input counts");
        require(number(result.stats,"output_samples")==samples+188,"Final output count");
        require(number(result.stats,"queued_input_samples")==0&&number(result.stats,"dropped_input_samples")==0,"Final queue/drop counts");
        require(engine.closes.get()==1,"Concurrent observation leaked model");waves(root,result,samples);
    }
    static void requestedBackend(Path root) throws Exception {
        Result result = new Result();
        try (EnhancementSession session = new EnhancementSession(root, root, root, result, () -> { }, 4,
                io.github.reuse.ReuseConfig.Backend.NNAPI_ACCELERATOR, true)) {
            Map<String,Object> stats = session.statistics();
            require("NNAPI_ACCELERATOR".equals(stats.get("requested_backend")), "Backend option was lost");
            require(Boolean.TRUE.equals(stats.get("nnapi_fp16_relaxation_requested")), "FP16 option was lost");
            require(Boolean.FALSE.equals(stats.get("nnapi_execution_verified")), "An unstarted session claimed execution");
            require(Boolean.FALSE.equals(stats.get("gpu_execution_verified")), "NNAPI registration claimed GPU proof");
        }
    }
    static void verificationFailure(Path root) throws Exception {
        Result result = new Result(); Identity engine = new Identity(); engine.failVerification = true;
        EnhancementSession session = new EnhancementSession(root, result, () -> { }, () -> engine, 8000);
        session.start(); await(result.done); session.close();
        require(result.ready.getCount() == 1, "Failed accelerator warm-up allowed microphone readiness");
        require("PROCESSING_ERROR".equals(result.error), "Verification failure was hidden");
        require(number(result.stats, "input_samples") == 0 && number(result.stats, "model_calls") == 0,
                "Verification failure became a recording");
        require(engine.closes.get() == 1, "Verification failure leaked the model");
    }
    public static void main(String[] args)throws Exception {
        String kind=args[0];Path dir=Path.of(args[1]);Files.createDirectories(dir);
        if(kind.equals("lengths"))lengths(dir);
        else if(kind.equals("overflow"))blocked(dir,true);
        else if(kind.equals("abort"))blocked(dir,false);
        else if(kind.equals("empty"))emptyStop(dir);
        else if(kind.equals("native"))nativeContracts();
        else if(kind.equals("snapshots"))concurrentSnapshots(dir);
        else if(kind.equals("backend"))requestedBackend(dir);
        else if(kind.equals("verification"))verificationFailure(dir);
        else failure(dir,kind);
        System.out.println("{\"case\":\""+kind+"\",\"passed\":true}");
    }
}
