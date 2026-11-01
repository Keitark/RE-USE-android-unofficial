# Original model work: Copyright (c) 2025 NVIDIA CORPORATION. All rights reserved.
# Model-specific adaptation: NVIDIA NSCLv1; see licenses/NVIDIA-NSCLv1.txt.
"""Optional proven rank4/channel64 LayerNorm -> NCHW -> PReLU fusion."""
from __future__ import annotations
import argparse,copy,hashlib,json,math,shutil
from pathlib import Path
import numpy as np
import onnx
from onnx import TensorProto,helper,numpy_helper
from optimize_dense_history import Constants,attrs,static_shapes,remove_dead_nodes,validate

def sha256(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def proof_shapes(model):
    analysis=copy.deepcopy(model);nodes=[];constants=Constants(analysis)
    for index,node in enumerate(analysis.graph.node):
        if node.domain=='com.reuse' and node.op_type=='FastSoftplus':
            nodes.append(helper.make_node('Identity',[node.input[0]],[node.output[0]],name=f'proof_softplus_{index}'));continue
        if node.domain=='com.reuse' and node.op_type=='TemporalConvSilu':
            nodes.extend([helper.make_node('Identity',[node.input[0]],[node.output[0]],name=f'proof_temporal_u_{index}'),
                          helper.make_node('Identity',[node.input[1]],[node.output[1]],name=f'proof_temporal_s_{index}')]);continue
        if node.domain in ('','ai.onnx') and node.op_type=='Pad':
            padding=np.asarray(constants.get(node.input[1]));name=f'__channel_norm_proof_pad_{index}'
            if padding.dtype!=np.int64 or padding.ndim!=1:raise ValueError('Cannot prove Pad shape')
            if any(v.name==name for v in analysis.graph.initializer):raise ValueError('Reserved proof name exists')
            analysis.graph.initializer.append(numpy_helper.from_array(padding,name));node.input[1]=name
        nodes.append(node)
    del analysis.graph.node[:];analysis.graph.node.extend(nodes)
    return static_shapes(analysis)

def optimize_graph(source,preserve_normalizations=None):
    if preserve_normalizations is None:preserve_normalizations=[]
    if not isinstance(preserve_normalizations,(list,tuple)) or any(not isinstance(n,str) for n in preserve_normalizations):
        raise ValueError('Preserved normalization names must be an explicit list')
    if len(set(preserve_normalizations))!=len(preserve_normalizations):raise ValueError('Duplicate preserved normalization name')
    model=copy.deepcopy(source);shapes=proof_shapes(model)
    consumers={}
    for node in model.graph.node:
        for i in node.input:consumers.setdefault(i,[]).append(node)
    weights={v.name:v for v in model.graph.initializer};public={v.name for v in model.graph.output}
    removed=set();replacements={};proofs=[]
    for norm in model.graph.node:
        if norm.domain not in ('','ai.onnx') or norm.op_type!='LayerNormalization':continue
        users=consumers.get(norm.output[0],[])
        if len(users)!=1 or users[0].op_type!='Transpose':continue
        transpose=users[0];users=consumers.get(transpose.output[0],[])
        if len(users)!=1 or users[0].op_type!='PRelu':continue
        prelu=users[0];shape=shapes.get(norm.input[0],[]);a=attrs(norm)
        if len(norm.input)!=3 or len(norm.output)!=1 or set(a)-{'axis','epsilon','stash_type'} or a.get('axis',-1) not in (-1,3) or a.get('stash_type',1)!=1:
            raise ValueError('Cannot prove float32 last-channel LayerNormalization')
        epsilon=a.get('epsilon',1e-5)
        if not math.isfinite(epsilon) or not 0<epsilon<=1:raise ValueError('Invalid normalization epsilon')
        if len(shape)!=4 or shape[-1]!=64 or any(not isinstance(d,int) or d<0 for d in shape):
            raise ValueError(f'Cannot prove static rank4/channel64 shape: {norm.name}: {shape}')
        if transpose.domain not in ('','ai.onnx') or attrs(transpose)!={'perm':[0,3,1,2]} or len(transpose.input)!=1 or len(transpose.output)!=1:
            raise ValueError('Cannot prove NHWC-to-NCHW transpose')
        if prelu.domain not in ('','ai.onnx') or prelu.attribute or len(prelu.input)!=2 or len(prelu.output)!=1 or prelu.input[0]!=transpose.output[0]:
            raise ValueError('Cannot prove PRelu contract')
        for name,expected in zip([norm.input[1],norm.input[2],prelu.input[1]],[[64],[64],[64,1,1]]):
            value=weights.get(name)
            if value is None or value.data_type!=TensorProto.FLOAT or list(value.dims)!=expected:raise ValueError('Expected fixed float32 scale/bias/slope weights')
        if norm.output[0] in public or transpose.output[0] in public:raise ValueError('Cannot remove public normalization outputs')
        if shapes.get(prelu.output[0])!=[shape[0],64,shape[1],shape[2]]:raise ValueError('Output layout proof failed')
        selected=[norm,transpose,prelu];names={n.name for n in selected}
        if len(names)!=3 or names&removed:raise ValueError('Overlapping fusion regions')
        removed.update(names)
        replacements[norm.name]=helper.make_node('ChannelNormPRelu',[norm.input[0],norm.input[1],norm.input[2],prelu.input[1]],
            list(prelu.output),name=f'/ChannelNormPRelu_{len(proofs)}',domain='com.reuse',epsilon=float(epsilon))
        proofs.append({'normalization':norm.name,'removed_nodes':sorted(names),'input_shape':shape,'output_shape':[shape[0],64,shape[1],shape[2]],'epsilon':epsilon})
    if len(proofs)!=18:raise ValueError(f'Expected exactly18 dense normalization regions, found {len(proofs)}')
    unknown=set(preserve_normalizations)-{p['normalization'] for p in proofs}
    if unknown:raise ValueError(f'Unknown preserved normalization names: {sorted(unknown)}')
    for proof in proofs:
        proof['fused']=proof['normalization'] not in preserve_normalizations
        if not proof['fused']:
            removed.difference_update(proof['removed_nodes'])
            del replacements[proof['normalization']]
    nodes=[]
    for node in model.graph.node:
        if node.name in replacements:nodes.append(replacements[node.name])
        if node.name not in removed:nodes.append(node)
    del model.graph.node[:];model.graph.node.extend(nodes);remove_dead_nodes(model)
    onnx.checker.check_model(model,full_check=True)
    if list(model.graph.input)!=list(source.graph.input) or list(model.graph.output)!=list(source.graph.output):raise ValueError('Public graph metadata changed')
    return model,proofs

def optimize(source_dir,output_dir,preserve_normalizations=None):
    if output_dir.exists():raise FileExistsError(f'Output exists: {output_dir}')
    manifest=json.loads((source_dir/'manifest.json').read_text());source=source_dir/'model.onnx'
    if sha256(source)!=manifest['model_sha256']:raise ValueError('Source hash mismatch')
    if manifest.get('sample_rate')!=8000 or manifest.get('frames_per_call')!=5 or manifest.get('exit_layers')!=3:raise ValueError('Only fixed8k/exit3/chunk5 is supported')
    if any(c['name']=='ChannelNormPRelu' for c in manifest['custom_ops']):raise ValueError('Source already fused')
    model,proofs=optimize_graph(onnx.load(source),preserve_normalizations);output_dir.mkdir(parents=True);graph=output_dir/'model.onnx';onnx.save(model,graph)
    fused=[p for p in proofs if p['fused']];preserved=[p['normalization'] for p in proofs if not p['fused']]
    previous=manifest.get('graph_optimizations');manifest['graph_optimizations']={'name':'channel_norm_prelu_v1','source_model_sha256':sha256(source),'proven_regions':proofs,'fused_regions':fused,'preserved_normalizations':preserved,'previous_optimizations':previous}
    manifest['custom_ops'].append({'domain':'com.reuse','name':'ChannelNormPRelu','version':1,'library':'reuse_ort_ops'})
    manifest['model_sha256']=sha256(graph);manifest['model_bytes']=graph.stat().st_size
    manifest['validation']={'phone_realtime_verified':False,'optimization_parity_verified':False,'scope':'Optional ChannelNormPRelu candidate; fresh recurrent parity required'}
    (output_dir/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');shutil.copy2(source_dir/'LICENSE-NVIDIA.txt',output_dir/'LICENSE-NVIDIA.txt')
    result={'source_model_sha256':sha256(source),'optimized_model_sha256':sha256(graph),'proven_regions':18,'fused_regions':len(fused),
            'fused_normalizations':[p['normalization'] for p in fused],'preserved_normalizations':preserved,
            'weights_and_public_io_unchanged':True,'scope':'Explicit layout preservation after proving all18 patterns; no timing claim'}
    (output_dir/'normalization-preservation-proof.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--custom-op-library',type=Path);p.add_argument('--speech-pcm',type=Path);p.add_argument('--validation-frames',type=int,default=40);p.add_argument('--validate-only',action='store_true')
    p.add_argument('--preserve-normalization',action='append',default=[],help='Exact proven normalization node to preserve; repeat explicitly, rejects unknown/duplicate names');a=p.parse_args()
    if a.validation_frames<2:p.error('At least two recurrent hops required')
    if not a.validate_only:print(json.dumps(optimize(a.source_dir,a.output_dir,a.preserve_normalization),indent=2))
    if a.custom_op_library:print(json.dumps(validate(a.source_dir,a.output_dir,a.custom_op_library,a.validation_frames,a.speech_pcm),indent=2))
    elif a.validate_only:p.error('--validate-only requires --custom-op-library')
