# Full neural GPU reconstruction

This experimental path reconstructs the full 8 kHz neural graph as standard
FP32 ONNX operations. It removes the CPU custom-operation dependencies from the
earlier GPU/CPU graph. **A successful host export or APK build does not establish
that the complete graph runs on Adreno, or that it runs faster.**

The separate **Test full neural GPU (experimental)** button disables CPU execution
provider fallback. Audio analysis, synthesis, recurrent-buffer handling and the
Android application still use the CPU. Here, “full neural GPU” refers to the
ONNX neural graph, not every instruction in the audio pipeline.

[Download the test APK and checksums](https://github.com/Keitark/RE-USE-android-unofficial/releases/tag/v0.6.0-full-gpu-probe1).
The [build receipt](validation/full-gpu-probe1-build.json) records 135 passing
Java tests, 41 Python tests and 223 package checks. Both app lint reports have
zero errors; device execution remains unverified for this APK.

## Why reconstruct the graph

The first device test passed a strict FP32 convolution on the direct QNN GPU
backend and a separate Vulkan arithmetic smoke test on Adreno 730. The original
full graph initialized but failed in its first warm-up at `TemporalConvSilu`.
That graph contains 45 custom nodes whose implementations explicitly select the
CPU execution provider. Fixing one tensor boundary would still leave a hybrid.

The new export uses the existing portable frequency scan and explicit five-hop
temporal recurrence in `tools/reuse_port.py` and `tools/reuse_chunk.py`. It keeps
the trained checkpoint, 8 kHz, exit 3, zero neural look-ahead, five 20 ms hops per
call, and all 21 FP32 recurrent states. No quantization or FP16 conversion is
introduced. Closed shape/constant expressions are folded before deployment;
changing a model equation or weakening a numerical limit is not an accepted fix.
The tree scan's one-pair base case omits an empty concatenation, avoiding
zero-length intermediate tensors without changing the recurrence.

Portable scans expose many more operations and intermediate tensors than the
fused CPU implementation. GPU compatibility and speed must therefore be
measured; removing custom nodes alone does not demonstrate an improvement.

The [host validation report](validation/full-gpu-standard-host.json) records
1,428 standard nodes, zero custom nodes, positive static runtime tensor shapes,
and unchanged trained initializers before constant folding. All 21 states were
checked with independent chronological feedback over 65 audio calls and one
flush. Against the original fused FP32 reference, waveform maximum absolute
error was `1.639e-7` and normalized RMSE was `6.597e-7`, within the existing
`0.001` / `0.005` limits. These are host CPU numerical results, not Adreno timing
or general audio-quality acceptance.

## Device test

Install the `0.6.0-full-gpu-probe1` compute-test APK and press **Test full neural
GPU (experimental)**. Keep the app visible. This button runs only the strict GPU
experiment and its original FP32 CPU reference, rather than the entire resource
matrix. Both request four ORT intra-op threads; the GPU neural graph must still
execute with CPU fallback disabled.

Compilation and three profiled warm-ups are separate from the measured public
audio sequence. The recurrent state is reset before the 65 chronological audio
calls and flush. A successful result requires:

- A hash-checked standard-ONNX candidate with a passing host sequence comparison.
- CPU fallback disabled, QNN-only warm-up execution evidence, and the actual
  loaded `GPU_QTI_AISW` backend, ID `4`.
- Complete finite output and unchanged whole-wave limits against the original
  FP32 CPU reference: maximum absolute error `0.001` and normalized RMSE `0.005`.

Unsupported operations, compilation errors, driver errors and numerical
failures remain explicit failures. There is no automatic CPU fallback or change
to the microphone demo. QNN graph/debug dumps and failed profiles are retained
privately where the runtime produces them. **Share diagnostics (no audio)**
exports the bounded report and backend log, not model dumps or audio.

The three-minute wall budget and Stop action are cooperative: native driver
calls cannot be forcibly interrupted. This is an unpaced public-clip experiment,
not microphone capture, sustained queue/playout validation or an audio-quality
acceptance test.

## Why four CPU threads improved

In the resource test, the original FP32 graph, weights, ORT 1.29 and native
library were identical across thread settings. Only the requested intra-op pool
size changed. Graph execution stayed sequential, inter-op threads stayed at one,
and thread spinning stayed disabled.

| FP32 CPU threads | Mean wall time per 100 ms chunk | Process CPU time per call |
| ---: | ---: | ---: |
| 2 | 48.707 ms | 92.300 ms |
| 4 | 37.934 ms | 120.577 ms |
| 6 | 40.603 ms | 138.992 ms |
| 8 | 45.452 ms | 162.962 ms |

Four threads reduced elapsed time by 22.12%, while process CPU time increased
30.64%. Existing ORT convolutions/projections and native kernels that use
`ParallelFor` can distribute independent work across the larger pool. The data
supports more overlapping CPU execution, not fewer calculations or a battery
saving. It does not establish exact active cores or affinity.

Six/eight threads were slower and consumed more process CPU time. Scheduling,
synchronization, cache/bandwidth contention and heterogeneous core placement
are possible causes; this run did not measure which caused the regression.
The four-thread result covers two short public-clip repetitions, with no
process call over 100 ms among 130 calls. A sustained paced test is still needed
before selecting a microphone default.

## Reproduce the assets and APK

Use the locked Python dependencies, JDK 17, SDK 35, NDK `26.1.10909125` and CMake
`3.22.1`. First prepare the existing eight-Conv fixture pack as described in
[the convolution experiments](qnn-convolution-experiments.md). Then run from the
repository root:

```powershell
.\.venv\Scripts\python.exe tools\export_model.py --source .research/upstream --output .research/gpu-full-standard-export --sample-rate 8000 --exit-layer 3 --frames-per-call 5 --scan-mode parallel --threads 2 --validation-frames 15
.\.venv\Scripts\python.exe tools\prepare_full_gpu.py --source .research/gpu-full-standard-export --fixtures .research/qnn-eightconv-assets --output .research/full-gpu-assets --custom-op-library native/build-windows-frequency-conv/Release/reuse_ort_ops.dll
.\gradlew.bat :qnn-probe:testDebugUnitTest :reuse-qnn:testDebugUnitTest :qnn-probe:assembleDebug :qnn-probe:lintDebug -PreuseNeonFma=true -PreuseScanN16Registers=true
```

The Windows host library is used only to validate the original fused reference.
The reconstructed graph has no custom-operation dependency. Source generation
and host numerical checks are separate from the final Android GPU acceptance.

NVIDIA weights and model-derived export code retain NSCLv1 noncommercial terms.
Qualcomm dependencies retain their AI Stack terms. This project is unofficial
and is not affiliated with or endorsed by NVIDIA or Qualcomm.
