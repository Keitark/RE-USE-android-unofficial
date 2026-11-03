package io.github.reuse;

import static org.junit.Assert.*;
import ai.onnxruntime.OrtProvider;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.Collections;
import java.util.EnumSet;
import org.junit.Test;

public class BackendSessionOptionsTest {
    @Test public void unsupportedAcceleratorRequestsFailInsteadOfBecomingCpu() throws Exception {
        ReuseConfig config = new ReuseConfig(8000, 3, 4,
                ReuseConfig.Backend.NNAPI_ACCELERATOR, false, Paths.get("private", "warmup"));
        assertThrows(IOException.class,
                () -> BackendSessionOptions.validateAccelerator(config, 28, EnumSet.of(OrtProvider.NNAPI)));
        assertThrows(IOException.class,
                () -> BackendSessionOptions.validateAccelerator(config, 29, EnumSet.of(OrtProvider.CPU)));
        BackendSessionOptions.validateAccelerator(config, 29, EnumSet.of(OrtProvider.CPU, OrtProvider.NNAPI));
        BackendSessionOptions.validateAccelerator(new ReuseConfig(8000), 26, Collections.emptySet());
    }

    @Test public void verificationCountsCompletedNnapiKernelsOnly() throws Exception {
        String events = "["
                + "{\"cat\":\"Node\",\"name\":\"NNAPI_0_kernel_time\",\"args\":{\"provider\":\"NnapiExecutionProvider\"}},"
                + "{\"cat\":\"Node\",\"name\":\"CPU_kernel_time\",\"args\":{\"provider\":\"CPUExecutionProvider\"}},"
                + "{\"cat\":\"Session\",\"name\":\"NNAPI_1_kernel_time\",\"args\":{\"provider\":\"NnapiExecutionProvider\"}},"
                + "{\"cat\":\"Node\",\"name\":\"NNAPI_fence_before\",\"args\":{\"provider\":\"NnapiExecutionProvider\"}},"
                + "{\"cat\":\"Node\",\"name\":\"Other_kernel_time\",\"args\":{\"provider\":\"QNNExecutionProvider\"}},"
                + "{\"cat\":\"Node\",\"name\":\"Missing_kernel_time\"},null]";
        assertEquals(1, BackendSessionOptions.countNnapiKernelEvents(events));
        assertEquals(0, BackendSessionOptions.countNnapiKernelEvents("[]"));
        assertThrows(IOException.class, () -> BackendSessionOptions.countNnapiKernelEvents("not json"));
    }
}
