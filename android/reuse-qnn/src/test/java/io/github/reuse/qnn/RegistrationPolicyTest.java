package io.github.reuse.qnn;

import static org.junit.Assert.*;
import java.io.IOException;
import org.junit.Test;

public final class RegistrationPolicyTest {
    @Test public void sameRuntimeAndRealPathShareRegistrationUntilFinalRelease() throws Exception {
        RegistrationPolicy policy = new RegistrationPolicy();
        Object environment = new Object();
        assertTrue(policy.acquire(environment, "/native/libqnn.so"));
        assertFalse(policy.acquire(environment, "/native/libqnn.so"));
        assertFalse(policy.release(environment, "/native/libqnn.so"));
        assertTrue(policy.release(environment, "/native/libqnn.so"));
        assertTrue(policy.acquire(environment, "/native/libqnn.so"));
        assertTrue(policy.release(environment, "/native/libqnn.so"));
    }

    @Test public void conflictingPathOrEnvironmentCannotStealAnActiveLease() throws Exception {
        RegistrationPolicy policy = new RegistrationPolicy();
        Object environment = new Object();
        policy.acquire(environment, "/native/libqnn.so");
        assertThrows(IOException.class, () -> policy.acquire(environment, "/other/libqnn.so"));
        assertThrows(IOException.class, () -> policy.acquire(new Object(), "/native/libqnn.so"));
        assertThrows(IllegalStateException.class, () -> policy.release(environment, "/other/libqnn.so"));
        assertTrue(policy.release(environment, "/native/libqnn.so"));
        assertThrows(IllegalStateException.class, () -> policy.release(environment, "/native/libqnn.so"));
    }

    @Test public void cleanupFailureBlocksUnsafeReRegistration() throws Exception {
        RegistrationPolicy policy = new RegistrationPolicy();
        Object environment = new Object();
        policy.acquire(environment, "/native/libqnn.so");
        policy.release(environment, "/native/libqnn.so");
        IOException cleanup = new IOException("native plugin still in use");
        policy.poison(cleanup);
        IOException failure = assertThrows(IOException.class, () -> policy.acquire(environment, "/native/libqnn.so"));
        assertSame(cleanup, failure.getCause());
    }
}
