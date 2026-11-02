package io.github.reuse.phone;

import android.content.Context;

/** CPU builds contain no optional QNN adapter or Qualcomm imports. */
final class PhoneSessionProviders {
    private PhoneSessionProviders() { }

    static PhoneCaptureController.SessionFactory optionalQnn(Context context) { return null; }
    static PhoneCaptureController.SessionFactory optionalQuantizedCpu(Context context) { return null; }
}
