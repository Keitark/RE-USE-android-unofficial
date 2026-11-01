package io.github.reuse;

/** Reusable unnormalised forward / normalised inverse FFT for N=5*2^k. */
final class MixedRadixFft {
    private final int size;
    private final int innerSize;
    private final int[] bitReverse;
    private final double[] real;
    private final double[] imag;
    private final double[] twiddleCos;
    private final double[] twiddleSin;

    MixedRadixFft(int size) {
        if (size < 10 || size % 5 != 0 || Integer.bitCount(size / 5) != 1) {
            throw new IllegalArgumentException("FFT size must be 5 times a power of two");
        }
        this.size = size;
        this.innerSize = size / 5;
        this.real = new double[size];
        this.imag = new double[size];
        this.twiddleCos = new double[size];
        this.twiddleSin = new double[size];
        this.bitReverse = new int[innerSize];
        int bits = Integer.numberOfTrailingZeros(innerSize);
        for (int i = 0; i < innerSize; i++) {
            bitReverse[i] = Integer.reverse(i) >>> (32 - bits);
        }
        for (int i = 0; i < size; i++) {
            twiddleCos[i] = Math.cos(2.0 * Math.PI * i / size);
            twiddleSin[i] = Math.sin(2.0 * Math.PI * i / size);
        }
    }

    void transform(double[] inputReal, double[] inputImag, boolean inverse) {
        if (inputReal.length != size || inputImag.length != size) {
            throw new IllegalArgumentException("FFT input length mismatch");
        }
        double sign = inverse ? 1.0 : -1.0;
        // Five independent radix-two FFTs of every fifth sample.
        for (int r = 0; r < 5; r++) {
            int base = r * innerSize;
            for (int m = 0; m < innerSize; m++) {
                int source = r + 5 * bitReverse[m];
                real[base + m] = inputReal[source];
                imag[base + m] = inputImag[source];
            }
            for (int len = 2; len <= innerSize; len <<= 1) {
                int half = len / 2;
                int stride = size / len;
                for (int block = 0; block < innerSize; block += len) {
                    for (int j = 0; j < half; j++) {
                        int even = base + block + j;
                        int odd = even + half;
                        int t = j * stride;
                        double c = twiddleCos[t], s = sign * twiddleSin[t];
                        double tr = c * real[odd] - s * imag[odd];
                        double ti = s * real[odd] + c * imag[odd];
                        real[odd] = real[even] - tr;
                        imag[odd] = imag[even] - ti;
                        real[even] += tr;
                        imag[even] += ti;
                    }
                }
            }
        }
        // Combine the five transforms. This is O(5*N), not a full DFT.
        double scale = inverse ? 1.0 / size : 1.0;
        for (int k = 0; k < size; k++) {
            double sumReal = 0.0, sumImag = 0.0;
            int j = k % innerSize;
            for (int r = 0; r < 5; r++) {
                int t = (r * k) % size;
                double c = twiddleCos[t], s = sign * twiddleSin[t];
                int source = r * innerSize + j;
                sumReal += c * real[source] - s * imag[source];
                sumImag += s * real[source] + c * imag[source];
            }
            inputReal[k] = sumReal * scale;
            inputImag[k] = sumImag * scale;
        }
    }
}
