package io.github.reuse;

import static org.junit.Assert.*;
import java.util.Random;
import org.junit.Test;

public class MixedRadixFftTest {
    @Test public void forwardMatchesIndependentDirectDftForComplexInput() {
        for (int size : new int[]{320, 640}) {
            Random random = new Random(99);
            double[] re = new double[size], im = new double[size];
            for (int i = 0; i < size; i++) { re[i] = random.nextDouble() - 0.5; im[i] = random.nextDouble() - 0.5; }
            double[] expectedRe = new double[size], expectedIm = new double[size];
            for (int k = 0; k < size; k++) {
                for (int n = 0; n < size; n++) {
                    double angle = -2.0 * Math.PI * k * n / size;
                    expectedRe[k] += re[n] * Math.cos(angle) - im[n] * Math.sin(angle);
                    expectedIm[k] += re[n] * Math.sin(angle) + im[n] * Math.cos(angle);
                }
            }
            new MixedRadixFft(size).transform(re, im, false);
            assertArrayEquals(expectedRe, re, 2e-10);
            assertArrayEquals(expectedIm, im, 2e-10);
        }
    }

    @Test public void inverseIsNormalizedAndPlanCanBeReused() {
        Random random = new Random(18);
        for (int size : new int[]{320, 640}) {
            MixedRadixFft fft = new MixedRadixFft(size);
            for (int run = 0; run < 4; run++) {
                double[] re = new double[size], im = new double[size];
                for (int i = 0; i < size; i++) { re[i] = random.nextDouble(); im[i] = random.nextDouble(); }
                double[] originalRe = re.clone(), originalIm = im.clone();
                fft.transform(re, im, false); fft.transform(re, im, true);
                assertArrayEquals(originalRe, re, 3e-14);
                assertArrayEquals(originalIm, im, 3e-14);
            }
        }
    }
}
