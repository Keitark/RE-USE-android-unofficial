"""Independent numerical/layout and strict-proof tests for ChannelNormPRelu."""
import os,sys
from pathlib import Path
import numpy as np
import onnx
from onnx import TensorProto,helper
import onnxruntime as ort
import pytest

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
from optimize_channel_norm import optimize_graph
LIBRARY=Path(os.environ.get('REUSE_CHANNEL_NORM_LIBRARY',ROOT/'native/build-windows-channel-norm/Release/reuse_ort_ops.dll'))

def session(shape,epsilon=1e-5,threads=2,scale_shape=(64,)):
    if not LIBRARY.is_file():pytest.skip('Build optional ChannelNormPRelu library first')
    n,h,w,c=shape
    g=helper.make_graph([helper.make_node('ChannelNormPRelu',['x','scale','bias','slope'],['y'],domain='com.reuse',epsilon=epsilon)],'norm',
        [helper.make_tensor_value_info('x',TensorProto.FLOAT,shape),helper.make_tensor_value_info('scale',TensorProto.FLOAT,list(scale_shape)),
         helper.make_tensor_value_info('bias',TensorProto.FLOAT,[64]),helper.make_tensor_value_info('slope',TensorProto.FLOAT,[64,1,1])],
        [helper.make_tensor_value_info('y',TensorProto.FLOAT,[n,c,h,w])])
    m=helper.make_model(g,opset_imports=[helper.make_opsetid('',17),helper.make_opsetid('com.reuse',1)]);m.ir_version=10
    o=ort.SessionOptions();o.intra_op_num_threads=threads;o.inter_op_num_threads=1;o.register_custom_ops_library(str(LIBRARY.resolve()))
    return ort.InferenceSession(m.SerializeToString(),o,providers=['CPUExecutionProvider'])

def feed(shape,seed=83,offset=0,spread=1):
    r=np.random.default_rng(seed)
    return {'x':(offset+r.normal(0,spread,shape)).astype(np.float32),'scale':r.uniform(-2,2,64).astype(np.float32),
            'bias':r.normal(0,.2,64).astype(np.float32),'slope':r.uniform(-.2,.8,(64,1,1)).astype(np.float32)}

def reference(values,epsilon=1e-5):
    x=values['x'].astype(np.float64);center=x-np.mean(x,axis=-1,keepdims=True)
    y=center/np.sqrt(np.mean(center*center,axis=-1,keepdims=True)+np.float32(epsilon))
    y=y*values['scale']+values['bias'];y=np.transpose(y,(0,3,1,2))
    return np.where(y<0,y*values['slope'],y).astype(np.float32)

@pytest.mark.parametrize('shape',[(1,1,1,64),(2,3,7,64),(1,5,163,64),(1,162,5,64),(0,5,2,64),(2,0,3,64)])
def test_numerics_layout_tiles_tails_empty_and_immutability(shape):
    values=feed(shape);before={k:v.copy() for k,v in values.items()}
    actual=session(shape).run(None,values)[0]
    np.testing.assert_allclose(actual,reference(values),rtol=1e-5,atol=2e-6)
    for k,v in values.items():np.testing.assert_array_equal(v,before[k])

@pytest.mark.parametrize('offset,spread',[(0,0),(7,0),(1e4,.01),(1,.00001),(-1e4,.01),(0,1e-7)])
def test_constant_and_low_variance_statistics(offset,spread):
    shape=(2,3,5,64);values=feed(shape,offset=offset,spread=spread)
    actual=session(shape).run(None,values)[0]
    np.testing.assert_allclose(actual,reference(values),rtol=1e-5,atol=2e-6)

def test_serial_parallel_are_identical():
    shape=(2,5,163,64);values=feed(shape)
    a=session(shape,threads=1).run(None,values)[0];b=session(shape,threads=4).run(None,values)[0]
    np.testing.assert_array_equal(a,b)

def test_nonfinite_row_propagates_nan_without_affecting_other_rows():
    shape=(1,2,3,64);values=feed(shape);values['x'][0,0,0,9]=np.nan;values['x'][0,0,1,2]=np.inf;values['x'][0,0,2,5]=-np.inf
    with np.errstate(invalid='ignore'):
        actual=session(shape).run(None,values)[0];expected=reference(values)
    np.testing.assert_allclose(actual,expected,rtol=1e-5,atol=2e-6,equal_nan=True)

def test_pinned_alias_rejected_before_mutation():
    shape=(1,3,7,64);values=feed(shape);before=values['x'].copy();s=session(shape);binding=s.io_binding()
    for k,v in values.items():binding.bind_cpu_input(k,v)
    binding.bind_output('y','cpu',0,np.float32,(1,64,3,7),values['x'].ctypes.data)
    with pytest.raises(RuntimeError,match='distinct input/output'):s.run_with_iobinding(binding)
    np.testing.assert_array_equal(values['x'],before)

@pytest.mark.parametrize('epsilon',[0.,-1e-5,float('nan'),float('inf')])
def test_invalid_epsilon_rejected(epsilon):
    with pytest.raises(Exception,match='epsilon'):session((1,2,3,64),epsilon=epsilon)

def source():
    p=ROOT/'models/reuse-8k-exit3-native-chunk5-sequence-pad-softplus-temporal/model.onnx'
    if not p.exists():pytest.skip('Frozen temporal graph is absent')
    return onnx.load(p)

def test_proves_exactly18_patterns_preserves_public_metadata_and_weights():
    original=source();candidate,proofs=optimize_graph(original)
    assert len(proofs)==18 and sum(n.op_type=='ChannelNormPRelu' for n in candidate.graph.node)==18
    assert sum(n.op_type=='LayerNormalization' for n in candidate.graph.node)==6
    assert list(candidate.graph.input)==list(original.graph.input) and list(candidate.graph.output)==list(original.graph.output)
    old={v.name:v.SerializeToString() for v in original.graph.initializer}
    assert all(old[v.name]==v.SerializeToString() for v in candidate.graph.initializer)

PRESERVED_FIVE=[
    '/dense_encoder/dense_conv_2/dense_conv_2.1/norm/LayerNormalization',
    '/mask_decoder_list.2/up_conv1/up_conv1.1/norm/LayerNormalization',
    '/mask_decoder_list.2/up_conv2.1/norm/LayerNormalization',
    '/phase_decoder_list.2/up_conv1/up_conv1.1/norm/LayerNormalization',
    '/phase_decoder_list.2/up_conv2.1/norm/LayerNormalization']

def test_explicit_preservation_proves18_and_fuses13_without_changing_io_weights():
    original=source();candidate,proofs=optimize_graph(original,PRESERVED_FIVE)
    assert len(proofs)==18 and sum(p['fused'] for p in proofs)==13
    assert {p['normalization'] for p in proofs if not p['fused']}==set(PRESERVED_FIVE)
    assert sum(n.op_type=='ChannelNormPRelu' for n in candidate.graph.node)==13
    assert sum(n.op_type=='LayerNormalization' for n in candidate.graph.node)==11
    assert set(PRESERVED_FIVE)<={n.name for n in candidate.graph.node}
    assert list(candidate.graph.input)==list(original.graph.input) and list(candidate.graph.output)==list(original.graph.output)
    old={v.name:v.SerializeToString() for v in original.graph.initializer}
    assert all(old[v.name]==v.SerializeToString() for v in candidate.graph.initializer)

@pytest.mark.parametrize('names',[['unknown'],[PRESERVED_FIVE[0],PRESERVED_FIVE[0]],PRESERVED_FIVE[0],[None]])
def test_preservation_rejects_unknown_duplicate_or_nonlist_names(names):
    with pytest.raises(ValueError):optimize_graph(source(),names)

def test_preserved_region_is_still_fully_proven_before_preservation():
    m=source();node=next(n for n in m.graph.node if n.name==PRESERVED_FIVE[0])
    next(a for a in node.attribute if a.name=='axis').i=2
    with pytest.raises(ValueError):optimize_graph(m,PRESERVED_FIVE)

@pytest.mark.parametrize('mutation',['axis','perm','slope','consumer','epsilon'])
def test_rejects_unproven_graph_contract(mutation):
    m=source();norm=next(n for n in m.graph.node if n.op_type=='LayerNormalization');users=lambda output:[n for n in m.graph.node if output in n.input]
    transpose=users(norm.output[0])[0];prelu=users(transpose.output[0])[0]
    if mutation=='axis':next(a for a in norm.attribute if a.name=='axis').i=2
    elif mutation=='perm':next(a for a in transpose.attribute if a.name=='perm').ints[:]=[0,2,1,3]
    elif mutation=='slope':next(v for v in m.graph.initializer if v.name==prelu.input[1]).dims[:]=[1,64,1]
    elif mutation=='epsilon':next(a for a in norm.attribute if a.name=='epsilon').f=0
    else:m.graph.node.append(helper.make_node('Identity',[norm.output[0]],['extra'],name='unproved_extra'))
    with pytest.raises((ValueError,onnx.shape_inference.InferenceError)):optimize_graph(m)
