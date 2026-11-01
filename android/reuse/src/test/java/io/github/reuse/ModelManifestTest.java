package io.github.reuse;

import static org.junit.Assert.*;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import org.json.JSONArray;
import org.json.JSONObject;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

public class ModelManifestTest {
    @Rule public TemporaryFolder temporary = new TemporaryFolder();

    @Test public void rejectsMixedRevisionModelAndConfiguration() throws Exception {
        Path model = temporary.newFile("model.onnx").toPath();
        Files.write(model, new byte[]{1,2,3,4});
        JSONObject manifest = manifest(model);
        Path metadata = write(manifest);
        assertEquals(1, ModelManifest.read(metadata, model, new ReuseConfig(8000)).states.size());
        assertThrows(IOException.class, () -> ModelManifest.read(metadata, model, new ReuseConfig(16000)));
        assertThrows(IOException.class, () -> ModelManifest.read(metadata, model, new ReuseConfig(8000,4,1)));
        Files.write(model, new byte[]{1,2,3,5});
        assertThrows(IOException.class, () -> ModelManifest.read(metadata, model, new ReuseConfig(8000)));
    }

    @Test public void rejectsDuplicateStatesUnboundedShapesAndLookahead() throws Exception {
        Path model = temporary.newFile("model.onnx").toPath();
        Files.write(model, new byte[]{1});
        JSONObject base = manifest(model);
        base.put("look_ahead_frames", 1);
        Path metadata = write(base);
        assertThrows(IOException.class, () -> ModelManifest.read(metadata, model, new ReuseConfig(8000)));
        base.put("look_ahead_frames", 0);
        base.getJSONArray("states").put(base.getJSONArray("states").getJSONObject(0));
        writeTo(metadata, base);
        assertThrows(IOException.class, () -> ModelManifest.read(metadata, model, new ReuseConfig(8000)));
        base.getJSONArray("states").remove(1);
        base.getJSONArray("states").getJSONObject(0).put("shape", new JSONArray(new long[]{99999999, 99999999}));
        writeTo(metadata, base);
        assertThrows(IOException.class, () -> ModelManifest.read(metadata, model, new ReuseConfig(8000)));
    }

    @Test public void customOperationIsExplicitAndWhitelistedWithoutAcceptingLibraryPaths() throws Exception {
        Path model = temporary.newFile("model.onnx").toPath();
        Files.write(model, new byte[]{1});
        JSONObject json = manifest(model);
        Path metadata = write(json);
        assertFalse(ModelManifest.read(metadata, model, new ReuseConfig(8000)).requiresSelectiveScan);
        JSONObject operation = new JSONObject().put("domain", "com.reuse").put("name", "SelectiveScan")
                .put("version", 1).put("library", "reuse_ort_ops");
        json.put("custom_ops", new JSONArray().put(operation));
        writeTo(metadata, json);
        assertTrue(ModelManifest.read(metadata, model, new ReuseConfig(8000)).requiresSelectiveScan);
        operation.put("library", "/sdcard/untrusted.so");
        writeTo(metadata, json);
        assertThrows(IOException.class, () -> ModelManifest.read(metadata, model, new ReuseConfig(8000)));
        operation.put("library", "reuse_ort_ops").put("version", 2);
        writeTo(metadata, json);
        assertThrows(IOException.class, () -> ModelManifest.read(metadata, model, new ReuseConfig(8000)));
    }

    @Test public void acceptsScanAndStepOnceEachButRejectsDuplicateUnknownOrOversizedLists() throws Exception {
        Path model = temporary.newFile("model.onnx").toPath();
        Files.write(model, new byte[]{1});
        JSONObject json = manifest(model);
        JSONObject scan = operation("SelectiveScan"), step = operation("SelectiveStep");
        Path metadata = write(json.put("custom_ops", new JSONArray().put(scan).put(step)));
        assertTrue(ModelManifest.read(metadata, model, new ReuseConfig(8000)).requiresSelectiveScan);
        writeTo(metadata, json.put("custom_ops", new JSONArray().put(step)));
        assertTrue(ModelManifest.read(metadata, model, new ReuseConfig(8000)).requiresSelectiveScan);
        writeTo(metadata, json.put("custom_ops", new JSONArray().put(scan).put(operation("SelectiveScan"))));
        assertThrows(IOException.class, () -> ModelManifest.read(metadata, model, new ReuseConfig(8000)));
        writeTo(metadata, json.put("custom_ops", new JSONArray().put(operation("Unknown"))));
        assertThrows(IOException.class, () -> ModelManifest.read(metadata, model, new ReuseConfig(8000)));
        writeTo(metadata, json.put("custom_ops", new JSONArray().put(scan).put(step).put(operation("SelectiveScan"))));
        assertThrows(IOException.class, () -> ModelManifest.read(metadata, model, new ReuseConfig(8000)));
    }

    @Test public void frameCountDefaultsToOneAndSupportsFixedFiveFrameChunks() throws Exception {
        Path model = temporary.newFile("model.onnx").toPath();
        Files.write(model, new byte[]{1});
        JSONObject json = manifest(model);
        Path metadata = write(json);
        ReuseConfig config = new ReuseConfig(8000);
        assertEquals(1, ModelManifest.read(metadata, model, config).framesPerCall);
        writeTo(metadata, json.put("frames_per_call", 1).put("chunk_size", 160));
        assertEquals(1, ModelManifest.read(metadata, model, config).framesPerCall);
        writeTo(metadata, json.put("frames_per_call", 5).put("chunk_size", 800));
        assertEquals(5, ModelManifest.read(metadata, model, config).framesPerCall);
        assertEquals(1, ModelManifest.read(metadata, model, config).states.size());
        for (Object invalid : new Object[]{0, -1, 2, 10, 5.5, "5", true}) {
            writeTo(metadata, json.put("frames_per_call", invalid));
            assertThrows(IOException.class, () -> ModelManifest.read(metadata, model, config));
        }
        json.put("frames_per_call", 5);
        for (Object invalid : new Object[]{799, 801, 800.5, "800"}) {
            writeTo(metadata, json.put("chunk_size", invalid));
            assertThrows(IOException.class, () -> ModelManifest.read(metadata, model, config));
        }
    }

    @Test public void acceptsFusedTemporalSequenceWithTheBundledLibraryOnly() throws Exception {
        Path model = temporary.newFile("model.onnx").toPath();
        Files.write(model, new byte[]{1});
        JSONObject sequence = operation("SelectiveSequence");
        JSONObject json = manifest(model).put("frames_per_call", 5).put("chunk_size", 800)
                .put("custom_ops", new JSONArray().put(operation("SelectiveScan")).put(sequence));
        Path metadata = write(json);
        assertTrue(ModelManifest.read(metadata, model, new ReuseConfig(8000)).requiresSelectiveScan);
        sequence.put("library", "some_other_library");
        writeTo(metadata, json);
        assertThrows(IOException.class, () -> ModelManifest.read(metadata, model, new ReuseConfig(8000)));
        sequence.put("library", "reuse_ort_ops");
        json.getJSONArray("custom_ops").put(operation("SelectiveSequence"));
        writeTo(metadata, json);
        assertThrows(IOException.class, () -> ModelManifest.read(metadata, model, new ReuseConfig(8000)));
    }

    @Test public void fastActivationMustUseTheBundledLibraryAndBeUnique() throws Exception {
        Path model = temporary.newFile("model.onnx").toPath();Files.write(model,new byte[]{1});
        JSONObject activation=operation("FastSoftplus");
        JSONObject json=manifest(model).put("custom_ops",new JSONArray()
                .put(operation("SelectiveScan")).put(operation("SelectiveSequence")).put(activation));
        Path metadata=write(json);
        assertTrue(ModelManifest.read(metadata,model,new ReuseConfig(8000)).requiresSelectiveScan);
        activation.put("library","different_library");writeTo(metadata,json);
        assertThrows(IOException.class,()->ModelManifest.read(metadata,model,new ReuseConfig(8000)));
        activation.put("library","reuse_ort_ops");json.getJSONArray("custom_ops").put(operation("FastSoftplus"));writeTo(metadata,json);
        assertThrows(IOException.class,()->ModelManifest.read(metadata,model,new ReuseConfig(8000)));
    }

    @Test public void temporalFusionUsesOnlyTheBundledV1Library() throws Exception {
        Path model = temporary.newFile("model.onnx").toPath();
        Files.write(model, new byte[]{1});
        JSONObject temporal = operation("TemporalConvSilu");
        JSONObject json = manifest(model).put("frames_per_call",5).put("chunk_size",800)
                .put("custom_ops",new JSONArray().put(operation("SelectiveScan"))
                        .put(operation("SelectiveStep")).put(operation("SelectiveSequence"))
                        .put(operation("FastSoftplus")).put(temporal));
        Path metadata=write(json);
        assertTrue(ModelManifest.read(metadata,model,new ReuseConfig(8000)).requiresSelectiveScan);
        temporal.put("version",2);writeTo(metadata,json);
        assertThrows(IOException.class,()->ModelManifest.read(metadata,model,new ReuseConfig(8000)));
        temporal.put("version",1).put("library","another_library");writeTo(metadata,json);
        assertThrows(IOException.class,()->ModelManifest.read(metadata,model,new ReuseConfig(8000)));
        temporal.put("library","reuse_ort_ops");json.getJSONArray("custom_ops").put(operation("TemporalConvSilu"));writeTo(metadata,json);
        assertThrows(IOException.class,()->ModelManifest.read(metadata,model,new ReuseConfig(8000)));
    }

    @Test public void additionalFusionsRequireUniqueBundledV1Operations() throws Exception {
        Path model = temporary.newFile("model.onnx").toPath();
        Files.write(model, new byte[]{1});
        for (String name : new String[]{"ChannelNormPRelu", "FrequencyConvSilu"}) {
            JSONObject selected = operation(name);
            JSONArray operations = new JSONArray();
            for (String other : new String[]{"SelectiveScan", "SelectiveStep", "SelectiveSequence",
                    "FastSoftplus", "TemporalConvSilu", "ChannelNormPRelu", "FrequencyConvSilu"}) {
                operations.put(other.equals(name) ? selected : operation(other));
            }
            JSONObject json = manifest(model).put("frames_per_call",5).put("chunk_size",800)
                    .put("custom_ops",operations);
            Path metadata=write(json);
            assertTrue(ModelManifest.read(metadata,model,new ReuseConfig(8000)).requiresSelectiveScan);
            selected.put("version",2);writeTo(metadata,json);
            assertThrows(IOException.class,()->ModelManifest.read(metadata,model,new ReuseConfig(8000)));
            selected.put("version",1).put("library","another_library");writeTo(metadata,json);
            assertThrows(IOException.class,()->ModelManifest.read(metadata,model,new ReuseConfig(8000)));
            selected.put("library","reuse_ort_ops");operations.put(operation(name));writeTo(metadata,json);
            assertThrows(IOException.class,()->ModelManifest.read(metadata,model,new ReuseConfig(8000)));
        }
    }

    @Test public void enhancersRejectTheWrongFrameContractBeforeLoadingRuntime() throws Exception {
        Path model = temporary.newFile("model.onnx").toPath();
        Files.write(model, new byte[]{1});
        JSONObject json = manifest(model);
        Path metadata = write(json);
        ReuseConfig config = new ReuseConfig(8000);
        IOException oneFrame = assertThrows(IOException.class,
                () -> new ChunkedReuseEnhancer(model, metadata, config));
        assertTrue(oneFrame.getMessage().contains("frames_per_call=5"));
        writeTo(metadata, json.put("frames_per_call", 5).put("chunk_size", 800));
        IOException fiveFrames = assertThrows(IOException.class,
                () -> new ReuseEnhancer(model, metadata, config));
        assertTrue(fiveFrames.getMessage().contains("ChunkedReuseEnhancer"));
    }

    private JSONObject operation(String name) throws Exception {
        return new JSONObject().put("domain", "com.reuse").put("name", name)
                .put("version", 1).put("library", "reuse_ort_ops");
    }

    private JSONObject manifest(Path model) throws Exception {
        StringBuilder hash = new StringBuilder();
        for (byte b : MessageDigest.getInstance("SHA-256").digest(Files.readAllBytes(model))) {
            hash.append(String.format("%02x", b & 255));
        }
        return new JSONObject().put("format_version",1).put("sample_rate",8000)
                .put("fft_size",320).put("hop_size",160).put("exit_layers",3)
                .put("look_ahead_frames",0).put("model_sha256",hash.toString())
                .put("states",new JSONArray().put(new JSONObject().put("input","state_000")
                        .put("output","next_state_000").put("shape",new JSONArray(new int[]{1,2,3}))));
    }

    private Path write(JSONObject json) throws Exception {
        Path path = temporary.newFile().toPath(); writeTo(path,json); return path;
    }
    private void writeTo(Path path, JSONObject json) throws Exception {
        Files.write(path,json.toString().getBytes(StandardCharsets.UTF_8));
    }
}
