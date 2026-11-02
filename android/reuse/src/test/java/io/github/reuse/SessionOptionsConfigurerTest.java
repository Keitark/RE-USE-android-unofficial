package io.github.reuse;

import static org.junit.Assert.*;
import java.io.IOException;
import java.nio.file.Paths;
import org.junit.Test;

public final class SessionOptionsConfigurerTest {
    @Test public void externalHookRequiresCpuConfigurationBeforeLoadingAnyModel() throws Exception {
        SessionOptionsConfigurer external = (environment, options) -> { };
        ReuseConfig cpu = new ReuseConfig(8000);
        ChunkedReuseEnhancer.validateExternalConfigurer(cpu, external);
        ReuseConfig nnapi = new ReuseConfig(8000, 3, 2, ReuseConfig.Backend.NNAPI_ACCELERATOR,
                false, Paths.get("private", "warmup"));
        ChunkedReuseEnhancer.validateExternalConfigurer(nnapi, null);
        assertThrows(IllegalArgumentException.class, () ->
                new ChunkedReuseEnhancer(null, null, nnapi, null, external));
    }

    @Test public void configurerCanPropagateCheckedConfigurationErrors() {
        SessionOptionsConfigurer external = (environment, options) -> { throw new IOException("backend unavailable"); };
        IOException failure = assertThrows(IOException.class, () -> external.configure(null, null));
        assertEquals("backend unavailable", failure.getMessage());
    }
}
