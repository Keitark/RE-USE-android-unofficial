"""Independent optional temporal Conv/SiLU/state tests and strict rewrite proof."""
import copy
import os
from pathlib import Path
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper
import onnxruntime as ort
from onnxruntime.capi.onnxruntime_pybind11_state import InvalidArgument
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from optimize_temporal_conv import optimize_graph
LIBRARY=Path(os.environ.get("REUSE_TEMPORAL_CONV_LIBRARY",ROOT/"native/build-windows-temporal-conv/Release/reuse_ort_ops.dll"))

def session(batch,time,channels,threads=2,state_channels=None):
    if not LIBRARY.is_file():pytest.skip("Build optional TemporalConvSilu library first")
    graph=helper.make_graph([helper.make_node("TemporalConvSilu",["u","state","weight","bias"],["y","next"],domain="com.reuse")],"temporal",
        [helper.make_tensor_value_info("u",TensorProto.FLOAT,[batch,time,channels]),
         helper.make_tensor_value_info("state",TensorProto.FLOAT,[batch,state_channels or channels,4]),
         helper.make_tensor_value_info("weight",TensorProto.FLOAT,[channels,1,4]),
         helper.make_tensor_value_info("bias",TensorProto.FLOAT,[channels])],
        [helper.make_tensor_value_info("y",TensorProto.FLOAT,[batch,time,channels]),
         helper.make_tensor_value_info("next",TensorProto.FLOAT,[batch,state_channels or channels,4])])
    model=helper.make_model(graph,opset_imports=[helper.make_opsetid("",17),helper.make_opsetid("com.reuse",1)]);model.ir_version=10
    options=ort.SessionOptions();options.intra_op_num_threads=threads;options.inter_op_num_threads=1
    options.register_custom_ops_library(str(LIBRARY.resolve()))
    return ort.InferenceSession(model.SerializeToString(),options,providers=["CPUExecutionProvider"])

def reference(u,state,weight,bias):
    history=np.concatenate((state,np.transpose(u,(0,2,1))),axis=2)
    values=np.zeros_like(u,dtype=np.float64)
    for t in range(u.shape[1]):
        for tap in range(4):values[:,t,:]+=history[:,:,1+t+tap].astype(np.float64)*weight[:,0,tap]
        values[:,t,:]+=bias
    e=np.exp(-np.abs(values))
    sigmoid=np.where(values>=0,1/(1+e),e/(1+e))
    return (values*sigmoid).astype(np.float32),history[:,:,-4:]

@pytest.mark.parametrize("shape",[(1,1,1),(2,1,7),(1,5,4),(2,5,65),(3,8,17),(81,5,256)])
def test_isolated_convolution_silu_and_exact_state_without_mutation(shape):
    b,t,c=shape;rng=np.random.default_rng(b+t+c)
    feed={"u":rng.normal(0,.5,shape).astype(np.float32),"state":rng.normal(0,.5,(b,c,4)).astype(np.float32),
          "weight":rng.normal(0,.3,(c,1,4)).astype(np.float32),"bias":rng.normal(0,.2,c).astype(np.float32)}
    before={k:v.copy() for k,v in feed.items()}
    actual=session(*shape).run(None,feed);expected=reference(**feed)
    np.testing.assert_allclose(actual[0],expected[0],rtol=3e-6,atol=2e-7)
    np.testing.assert_array_equal(actual[1],expected[1])
    for key in feed:np.testing.assert_array_equal(feed[key],before[key])

def test_serial_and_parallel_outputs_are_identical():
    rng=np.random.default_rng(73);shape=(81,5,256)
    feed={"u":rng.normal(size=shape).astype(np.float32),"state":rng.normal(size=(81,256,4)).astype(np.float32),
          "weight":rng.normal(size=(256,1,4)).astype(np.float32),"bias":rng.normal(size=256).astype(np.float32)}
    a=session(*shape,threads=1).run(None,feed);b=session(*shape,threads=4).run(None,feed)
    for x,y in zip(a,b):np.testing.assert_array_equal(x,y)

def test_silu_dense_range_subnormals_and_nonfinite_semantics():
    bias=np.concatenate((np.linspace(-80,80,4097,dtype=np.float32),np.asarray([-100,-np.inf,np.inf,np.nan],np.float32)))
    c=len(bias);feed={"u":np.zeros((1,1,c),np.float32),"state":np.zeros((1,c,4),np.float32),
        "weight":np.zeros((c,1,4),np.float32),"bias":bias}
    with np.errstate(invalid="ignore",under="ignore"):
        actual=session(1,1,c).run(None,feed)[0][0,0]
        expected=bias.astype(np.float64)*np.where(bias>=0,1/(1+np.exp(-np.abs(bias.astype(np.float64)))),np.exp(-np.abs(bias.astype(np.float64)))/(1+np.exp(-np.abs(bias.astype(np.float64)))))
    np.testing.assert_allclose(actual[:4097],expected[:4097],rtol=2e-6,atol=1e-37)
    assert abs(float(actual[-4])-float(expected[-4]))<1e-37
    assert np.isnan(actual[-3]) and np.isposinf(actual[-2]) and np.isnan(actual[-1])

def test_2048_updates_and_reset_keep_exact_history():
    rng=np.random.default_rng(2048);shape=(2,1,7);s=session(*shape,threads=1)
    state=rng.normal(size=(2,7,4)).astype(np.float32);initial=state.copy()
    weight=rng.normal(0,.1,(7,1,4)).astype(np.float32);bias=rng.normal(0,.1,7).astype(np.float32)
    first=None
    for iteration in range(2048):
        u=rng.normal(0,.2,shape).astype(np.float32);before=state.copy()
        actual=s.run(None,{"u":u,"state":state,"weight":weight,"bias":bias});expected=reference(u,state,weight,bias)
        np.testing.assert_allclose(actual[0],expected[0],rtol=3e-6,atol=2e-7)
        np.testing.assert_array_equal(actual[1],expected[1]);np.testing.assert_array_equal(state,before)
        if first is None:first=(u.copy(),actual)
        state=actual[1]
    reset=s.run(None,{"u":first[0],"state":initial,"weight":weight,"bias":bias})
    for a,b in zip(reset,first[1]):np.testing.assert_array_equal(a,b)

@pytest.mark.parametrize("output_alias",["u","state"])
def test_pinned_input_alias_rejected_before_any_mutation(output_alias):
    rng=np.random.default_rng(42);shape=(1,5,4)
    feed={"u":rng.normal(size=shape).astype(np.float32),"state":rng.normal(size=(1,4,4)).astype(np.float32),
          "weight":rng.normal(size=(4,1,4)).astype(np.float32),"bias":np.zeros(4,np.float32)}
    before={k:v.copy() for k,v in feed.items()};s=session(*shape);binding=s.io_binding()
    for k,v in feed.items():binding.bind_cpu_input(k,v)
    if output_alias=="u":
        binding.bind_output("y","cpu",0,np.float32,shape,feed["u"].ctypes.data);binding.bind_output("next","cpu")
    else:
        binding.bind_output("y","cpu");binding.bind_output("next","cpu",0,np.float32,(1,4,4),feed["state"].ctypes.data)
    with pytest.raises(RuntimeError,match="distinct input and output"):s.run_with_iobinding(binding)
    for k,v in feed.items():np.testing.assert_array_equal(v,before[k])

def test_bad_state_shape_rejected():
    s=session(1,5,4,state_channels=3)
    with pytest.raises(InvalidArgument,match="requires u"):
        s.run(None,{"u":np.zeros((1,5,4),np.float32),"state":np.zeros((1,3,4),np.float32),
                    "weight":np.zeros((4,1,4),np.float32),"bias":np.zeros(4,np.float32)})

def test_pinned_output_overlap_is_rejected():
    shape=(1,5,4);s=session(*shape);binding=s.io_binding()
    feed={"u":np.zeros(shape,np.float32),"state":np.zeros((1,4,4),np.float32),
          "weight":np.zeros((4,1,4),np.float32),"bias":np.zeros(4,np.float32)}
    for k,v in feed.items():binding.bind_cpu_input(k,v)
    shared=np.full(shape,17,np.float32);before=shared.copy()
    binding.bind_output("y","cpu",0,np.float32,shape,shared.ctypes.data)
    binding.bind_output("next","cpu",0,np.float32,(1,4,4),shared.ctypes.data)
    with pytest.raises(RuntimeError,match="distinct output"):s.run_with_iobinding(binding)
    np.testing.assert_array_equal(shared,before)

def source_model():
    path=ROOT/"models/reuse-8k-exit3-native-chunk5-sequence-pad-softplus/model.onnx"
    if not path.exists():pytest.skip("Frozen source model is not present")
    return onnx.load(path)

def test_rewrite_proves_three_regions_and_keeps_public_contract():
    original=source_model();candidate,proofs=optimize_graph(original)
    assert len(proofs)==3
    assert sum(n.op_type=="TemporalConvSilu" for n in candidate.graph.node)==3
    assert sum(n.op_type=="Conv" for n in original.graph.node)-sum(n.op_type=="Conv" for n in candidate.graph.node)==3
    assert original.graph.input==candidate.graph.input and original.graph.output==candidate.graph.output

@pytest.mark.parametrize("mutation",["dilation","extra_consumer","weight_shape","state_binding","slice_axis"])
def test_rewrite_rejects_contract_changes(mutation):
    model=source_model();nodes={n.name:n for n in model.graph.node}
    if mutation=="dilation":
        next(a for a in nodes['/Conv'].attribute if a.name=='dilations').ints[:]=[2]
    elif mutation=="extra_consumer":
        model.graph.node.append(helper.make_node("Identity",[nodes['/Conv'].output[0]],["unexpected_consumer"],name="extra"))
    elif mutation=="weight_shape":
        name=nodes['/Conv'].input[1];weight=next(v for v in model.graph.initializer if v.name==name);weight.dims[:]=[256,4,1]
    elif mutation=="state_binding":nodes['/Concat_3'].input[0]='state_007'
    else:
        # Independent consumer of the constant avoids mutating any other Slice.
        axis=helper.make_node("Constant",[],["wrong_axis"],name="wrong_axis_constant",value=helper.make_tensor("a",TensorProto.INT64,[1],[1]))
        model.graph.node.insert(0,axis);nodes['/Slice_4'].input[3]='wrong_axis'
    with pytest.raises((ValueError,onnx.shape_inference.InferenceError)):optimize_graph(model)
