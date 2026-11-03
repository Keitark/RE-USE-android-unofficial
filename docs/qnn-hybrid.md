# Experimental QNN hybrid RE-USE

This page records the initial **one-convolution** candidate. The current
experimental APK uses eight quantized convolutions; see its
[results, download and build guide](qnn-convolution-experiments.md).

The complete causal RE-USE pipeline now runs with CPU and Qualcomm Hexagon NPU
execution on Xperia 1 IV. In the current **unprofiled** foreground comparison,
CPU execution of the quantized candidate was fastest. The hybrid result
establishes execution; it does not establish a useful NPU speed advantage over
the same quantized graph on CPU.

The [sanitized device receipt](validation/qnn-xperia-hybrid.json) contains the
model hashes, numerical gates, sample accounting and provider evidence. The
[QNN probe guide](qnn-probe.md) covers the separate layer tests, runtime
dependencies and licenses. These experiments are unofficial and use NVIDIA
model-derived weights under NSCLv1 for noncommercial research and education.

## Complete pipeline result

Sony SO-51C / Snapdragon 8 Gen 1 (SM8450, Hexagon V69), Android 14; ONNX Runtime
1.29.0, QNN plugin 2.7.0 and QNN runtime 2.51.0. All backends use two CPU
intra-op threads; the hybrid uses balanced HTP mode with FP16 relaxation
disabled.

The packaged public NVIDIA clip contains **6.48 seconds / 51,840 samples** at
8 kHz. Each backend processes 65 chronological 100 ms calls, plus a flush,
after three warm-ups. All 21 FP32 recurrent states advance in separate buffers
per backend. Exit 3, zero look-ahead and five chronological hops are retained.

| Complete processing mode | Mean call | p95 call | Offline processing RTF |
| --- | ---: | ---: | ---: |
| Original FP32 graph on CPU | 69.73 ms | 75.33 ms | 0.7097 |
| Single-Conv quantized candidate on CPU | **63.11 ms** | **65.26 ms** | **0.6434** |
| Same candidate with QNN HTP + CPU | 67.06 ms | 70.58 ms | 0.6828 |

**The measured calls and flush are unprofiled.** Provider profiling covers only
three preceding zero-input warm-ups, then stops before state reset and the
measured sequence. Timings measure Java analysis/synthesis and synchronous
inference, excluding compilation, warm-up, profile verification, comparisons
and file writes. Mean/p95 describe the 65 process calls; RTF also includes the flush and
divides by the real 6.48-second input duration. The screen was interactive,
device idle was false and power saving was off. Backend order, CPU clocks and
thermal conditions were not controlled.

The earlier profiling-on means were **78.27 ms CPU FP32, 72.25 ms CPU QDQ and
73.81 ms hybrid**. The receipt preserves that separate run. These sequential
trials do not isolate profiling overhead from clock or thermal variation.

All three modes returned the complete trimmed input length with finite output.
None of the 65 process calls exceeded 100 ms in this run. This was offline,
unpaced processing: no microphone, live queue, speaker playout or underrun
measurement was involved. It does not establish sustained real-time behavior.

## Paced 16 kHz PCM pipeline

A separate foreground run feeds the same public clip through the actual
`EnhancementSession` pipeline: 800 PCM16 samples every 50 ms, the production
16→8→16 kHz FIR, 500 ms bounded queue, model, and stop/drain lifecycle.
All three modes accepted and processed **103,680 input samples**, produced
**103,868 output samples** including the 188-sample FIR tail, and reported
zero drops, sequence gaps, queue overflows and 100 ms model-call overruns.

| Mode | Mean model call | Processing RTF | Maximum queued input | Maximum queue age |
| --- | ---: | ---: | ---: | ---: |
| FP32 CPU | 68.26 ms | 0.7108 | 50 ms | 27.44 ms |
| Partially quantized CPU | 67.55 ms | 0.6948 | 80 ms | 37.21 ms |
| QNN + CPU | 72.32 ms | 0.7423 | 50 ms | 44.09 ms |

The CPU quantization gain was very small in this paced run. QNN was slower.
These sequential, short tests do not control thermal state or clocks. Processing
RTF includes model calls, flush and FIR computation, divided by input duration;
it excludes pacing waits and initialization. Model-call means exclude FIR.
QNN startup independently showed 12 partitions, each executing three times,
plus CPU work; input processing ran without profiling.

The 16 kHz fixture is deterministic interpolation of the public 8 kHz clip.
It tests pipeline mechanics, not microphone hardware, wider-band enhancement,
speech quality, sustained operation or `AudioTrack` playout/underruns. Maximum
queue occupancy and queue age measure different things and do not equal total
input-to-output latency. Full values are in the sanitized receipt.

## Selected graph and rejected alternatives

The selected candidate replaces only
`/dense_encoder/dense_block/dense_block.3.1/Conv` with the already tested UINT8
QDQ convolution and INT32 bias. Original normalization, PReLU, Mamba operations,
other initializers and all 21 state definitions remain unchanged in the source
graph. The FP32 replacement control is numerically identical on the public
host reference sequence.

Runtime placement is broader than that one graph edit. The current run's three
zero-input warm-ups contain **36 QNN and 1,092 CPU kernel events**, with **12
QNN partitions per warm-up**. All three matching warm-up profiles were
independently read. The subsequent measured audio calls and flush have no
provider trace.
The same static session continues after profiling ends and all states reset.

The earlier fully profiled hybrid run contains **828 QNN kernel events across
69 model runs: 12 QNN partitions per run**, alongside **25,116 CPU kernel
events**. Its three full profiles were independently read as well. These
records verify mixed execution through QNN's HTP path; partition/event counts
do not measure NPU compute share or hardware utilization.

Two larger encoder-tail candidates combined both convolutions with lowered
normalization/PReLU. They were rejected by their predefined numerical gates:

| Grouped activation precision | Maximum error versus CPU QDQ | Predeclared limit | Elements outside limit |
| --- | ---: | ---: | ---: |
| UINT8 | 0.076355 | 0.053448 | 5 |
| UINT16, UINT8 weights | 0.009705 | 0.000208 | 11,857 |

These are QNN-versus-CPU errors for the same quantized graph, distinct from
quantization error relative to original FP32. Neither grouped candidate was
accepted for full hybrid use, and no tolerance was relaxed to obtain a pass.

The selected single-Conv candidate's complete CPU output differs from FP32 by
waveform RMSE **0.000320** on this public clip. Hybrid versus CPU execution of
the quantized candidate has RMSE **0.000139**. These are reference waveform
differences, without a perceptual quality acceptance threshold. Calibration and
testing on one clip do not validate general speech quality.

## Smartphone microphone demo

The experimental phone app is **RE-USE QNN Demo (Unofficial)**, package
`io.github.reuse.phone.qnn`, version `0.4.0-qnn-mic-preview1`. It installs
alongside the original CPU demo. After generating the selected model assets,
build it with:

```powershell
.\gradlew.bat :phone-demo:assembleDebug -PreuseQnnExperimental=true '-PreuseQnnPhoneModel=.research/qnn-conv-hybrid-assets/full-candidate' -PreuseNeonFma=true -PreuseScanN16Registers=true
```

The three selectable modes are **CPU · original FP32**, **CPU · quantized
convolutions**, and **QNN NPU + CPU · quantized convolutions**. Choose a
mode, press **Start recording**, then **Stop & save**. QNN mode verifies provider
execution during warm-up before opening the microphone. Use **Share diagnostics**
for queue, drop and processing statistics.

The phone flow captures 16 kHz audio and resamples to the 8 kHz model and back.
The first table measures the 8 kHz public-clip enhancer. The paced test above
validates the shared FIR/queue path on public data; actual microphone capture
has not been measured with this QNN APK. A saved enhanced WAV joins produced PCM, so
its sample continuity alone does not establish uninterrupted live playout.

## Run the public-clip benchmark

Use the hybrid build from `:qnn-probe`, package `io.github.reuse.qnnprobe`.
The older probe1/probe2 release APKs contain isolated layer tests only.

1. Open the experimental app with the Xperia unlocked and the screen visible.
2. Press **Run convolution + streaming tests** for the isolated prerequisite and
   complete streaming comparison.
3. Keep the app in the foreground until all CPU and hybrid results finish.
4. Use **Share diagnostics** to export JSON, or copy the diagnostic text.

The benchmark app processes its packaged public test clip and does not request microphone
or Internet access. Initialization or numerical failures are retained in the
diagnostics; provider detection alone is not an execution pass.

The additional paced PCM test can be started over an authorized ADB connection:

```powershell
adb shell am start -S -W -n io.github.reuse.qnnprobe/.QnnProbeActivity --ez keep_screen_on true --ez run_tests true --ez run_pcm_tests true
```

Unlock the device first. This runs the prerequisite, full enhancer comparison,
and three paced PCM sessions; it never opens the microphone. Its
`public-pcm-benchmark.json` result is included in the final shared diagnostics.

For a source build, first prepare the pinned model, public sample and host
custom-op library using the [build guide](qnn-probe.md#build-from-source).
Generate probe2 fixtures if they are not already present, then the selected
hybrid assets, before building:

```powershell
.venv\Scripts\python.exe tools\prepare_qnn_probe.py --custom-op-library native/build-qnn-reference/Release/reuse_ort_ops.dll
.venv\Scripts\python.exe tools\prepare_qnn_conv_hybrid.py --custom-op-library native/build-qnn-reference/Release/reuse_ort_ops.dll
.\gradlew.bat :qnn-probe:assembleDebug '-PreuseQnnProbeAssets=.research/qnn-conv-hybrid-assets' -PreuseNeonFma=true -PreuseScanN16Registers=true
```

The generator preserves existing output directories. Use a fresh output when
repeating generation. The default selected assets are under
`.research/qnn-conv-hybrid-assets/`; generated models and public audio fixtures
are copied into the APK and are not checked into Git.

Further timing should repeat the same-device unprofiled comparison.
Microphone validation, sustained operation and
representative quality evaluation remain separate work.
