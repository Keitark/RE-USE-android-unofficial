# Original model work: Copyright (c) 2025 NVIDIA CORPORATION. All rights reserved.
# Model-specific adaptation: NVIDIA NSCLv1; see licenses/NVIDIA-NSCLv1.txt.
"""Opt-in, strictly proven four-tap temporal Conv/SiLU/state-layout fusion."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from optimize_dense_history import Constants, attrs, remove_dead_nodes, static_shapes, validate

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def optimize_graph(source):
    model=copy.deepcopy(source)
    analysis=copy.deepcopy(model)
    # Analysis-only metadata contract for the existing unary float operation.
    for node in analysis.graph.node:
        if node.domain=="com.reuse" and node.op_type=="FastSoftplus":
            if len(node.input)!=1 or len(node.output)!=1 or node.attribute:
                raise ValueError("Invalid FastSoftplus proof contract")
            node.domain="";node.op_type="Identity"
    constants_for_analysis=Constants(analysis)
    for index,node in enumerate(analysis.graph.node):
        if node.domain in ("","ai.onnx") and node.op_type=="Pad" and len(node.input)>1:
            padding=np.asarray(constants_for_analysis.get(node.input[1]))
            if padding.dtype!=np.int64 or padding.ndim!=1:raise ValueError("Cannot prove Pad shape")
            name=f"__temporal_analysis_pad_{index}"
            if any(v.name==name for v in analysis.graph.initializer):raise ValueError("Reserved proof name exists")
            analysis.graph.initializer.append(numpy_helper.from_array(padding,name));node.input[1]=name
    shapes=static_shapes(analysis)
    producers={out:node for node in model.graph.node for out in node.output}
    consumers={}
    for node in model.graph.node:
        for name in node.input:consumers.setdefault(name,[]).append(node)
    initializers={item.name:item for item in model.graph.initializer}
    outputs={item.name for item in model.graph.output}
    constants=Constants(model)
    removed=set(); replacements={}; proofs=[]

    def standard(node,kind):
        if node is None or node.domain not in ("","ai.onnx") or node.op_type!=kind or len(node.output)!=1:
            raise ValueError(f"Cannot prove standard {kind} node")
        return node
    def parent(name,kind):return standard(producers.get(name),kind)
    def only(name,expected):
        actual=[node.name for node in consumers.get(name,[])]
        if sorted(actual)!=sorted(expected):raise ValueError(f"Unexpected consumers of {name}: {actual}")
    def slice_indices(node,length):
        standard(node,"Slice")
        if node.attribute or len(node.input) not in (4,5):raise ValueError("Unexpected Slice contract")
        values=[np.asarray(constants.get(name)) for name in node.input[1:]]
        if any(v.dtype!=np.int64 or v.size!=1 or v.ndim!=1 for v in values):
            raise ValueError("Slice bounds must be scalar int64 vectors")
        start,end,axis=[int(v[0]) for v in values[:3]]
        step=int(values[3][0]) if len(values)==4 else 1
        if axis!=2 or step!=1:raise ValueError("Temporal history must slice axis2 with unit stride")
        return list(range(length))[slice(start,end,step)]

    for layer in range(3):
        conv_name="/Conv"+(f"_{layer}" if layer else "")
        conv=next((node for node in model.graph.node if node.name==conv_name),None)
        standard(conv,"Conv")
        if len(conv.input)!=3 or attrs(conv)!={"dilations":[1],"group":256,"kernel_shape":[4],"pads":[0,0],"strides":[1]}:
            raise ValueError(f"Unexpected temporal Conv attributes: {conv_name}")
        weight,bias=[initializers.get(name) for name in conv.input[1:]]
        if weight is None or bias is None or weight.data_type!=TensorProto.FLOAT or bias.data_type!=TensorProto.FLOAT or list(weight.dims)!=[256,1,4] or list(bias.dims)!=[256]:
            raise ValueError("Temporal Conv must have fixed float32 weights[256,1,4]/bias[256]")
        trim=parent(conv.input[0],"Slice");history=parent(trim.input[0],"Concat")
        if len(history.input)!=2 or attrs(history) not in ({"axis":2},{"axis":-1}):raise ValueError("Unexpected history concat")
        state,current_nt=history.input
        expected_state=f"state_{5+2*layer:03d}";expected_next=f"next_state_{5+2*layer:03d}"
        if state!=expected_state:raise ValueError("Temporal state binding changed")
        pre=parent(current_nt,"Transpose")
        if attrs(pre)!={"perm":[0,2,1]} or len(pre.input)!=1:raise ValueError("Unexpected input transpose")
        current=pre.input[0]
        if shapes.get(state)!=[81,256,4] or shapes.get(current)!=[81,5,256] or shapes.get(history.output[0])!=[81,256,9]:
            raise ValueError("Cannot prove fixed B81/T5/I256 temporal shapes")
        if slice_indices(trim,9)!=list(range(1,9)):raise ValueError("Conv does not use last3 history + current5")
        shift=parent(expected_next,"Slice")
        if expected_next not in outputs or shift.input[0]!=history.output[0] or slice_indices(shift,9)!=[5,6,7,8]:
            raise ValueError("Cannot prove next convolution state is exact final4 inputs")
        sigmoid=next((n for n in consumers.get(conv.output[0],[]) if n.op_type=="Sigmoid"),None)
        standard(sigmoid,"Sigmoid")
        mul=next((n for n in consumers.get(conv.output[0],[]) if n.op_type=="Mul"),None)
        standard(mul,"Mul")
        if sigmoid.attribute or sigmoid.input!=[conv.output[0]] or mul.attribute or sorted(mul.input)!=sorted([conv.output[0],sigmoid.output[0]]):
            raise ValueError("Cannot prove exact SiLU expression")
        users=consumers.get(mul.output[0],[])
        if len(users)!=1:raise ValueError("Unexpected SiLU consumers")
        post=standard(users[0],"Transpose")
        if attrs(post)!={"perm":[0,2,1]} or shapes.get(post.output[0])!=[81,5,256]:
            raise ValueError("Unexpected activation output transpose")
        only(pre.output[0],[history.name]);only(history.output[0],[shift.name,trim.name])
        only(trim.output[0],[conv.name]);only(conv.output[0],[sigmoid.name,mul.name])
        only(sigmoid.output[0],[mul.name]);only(mul.output[0],[post.name]);only(expected_next,[])
        selected=[pre,history,trim,shift,conv,sigmoid,mul,post]
        names={n.name for n in selected}
        if len(names)!=8 or names&removed:raise ValueError("Overlapping or unnamed fusion region")
        internal={out for n in selected for out in n.output}-{post.output[0],expected_next}
        if internal&outputs:raise ValueError("Fusion would remove a public output")
        removed.update(names)
        replacements[pre.name]=helper.make_node("TemporalConvSilu",[current,state,conv.input[1],conv.input[2]],
            [post.output[0],expected_next],name=f"/TemporalConvSilu_{layer}",domain="com.reuse")
        proofs.append({"layer":layer,"state":state,"next_state":expected_next,"input_shape":[81,5,256],
                       "state_shape":[81,256,4],"removed_nodes":sorted(names)})
    if len(proofs)!=3:raise ValueError("Expected exactly three temporal preludes")
    nodes=[]
    for node in model.graph.node:
        if node.name in replacements:nodes.append(replacements[node.name])
        if node.name not in removed:nodes.append(node)
    del model.graph.node[:];model.graph.node.extend(nodes)
    remove_dead_nodes(model)
    onnx.checker.check_model(model,full_check=True)
    if [v.SerializeToString() for v in source.graph.input]!=[v.SerializeToString() for v in model.graph.input] or [v.SerializeToString() for v in source.graph.output]!=[v.SerializeToString() for v in model.graph.output]:
        raise ValueError("Fusion changed graph input/output metadata")
    return model,proofs

def optimize(source_dir,output_dir):
    if output_dir.exists():raise FileExistsError(f"Output exists: {output_dir}")
    manifest=json.loads((source_dir/"manifest.json").read_text())
    source=source_dir/"model.onnx"
    if sha256(source)!=manifest["model_sha256"]:raise ValueError("Source model hash mismatch")
    if manifest.get("sample_rate")!=8000 or manifest.get("frames_per_call")!=5 or manifest.get("exit_layers")!=3:
        raise ValueError("Only fixed 8k/exit3/chunk5 graphs are supported")
    if any(item["name"]=="TemporalConvSilu" for item in manifest.get("custom_ops",[])):
        raise ValueError("Source already requests TemporalConvSilu")
    model,proofs=optimize_graph(onnx.load(source))
    output_dir.mkdir(parents=True)
    graph=output_dir/"model.onnx";onnx.save(model,graph)
    previous=manifest.get("graph_optimizations")
    manifest["graph_optimizations"]={"name":"temporal_conv_silu_v1","source_model_sha256":sha256(source),
        "fused_temporal_preludes":3,"proofs":proofs,"previous_optimizations":previous}
    manifest["custom_ops"].append({"domain":"com.reuse","name":"TemporalConvSilu","version":1,"library":"reuse_ort_ops"})
    manifest["model_sha256"]=sha256(graph);manifest["model_bytes"]=graph.stat().st_size
    manifest["validation"]={"phone_realtime_verified":False,"optimization_parity_verified":False,
                            "scope":"Optional TemporalConvSilu candidate; fresh recurrent parity required"}
    (output_dir/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    shutil.copy2(source_dir/"LICENSE-NVIDIA.txt",output_dir/"LICENSE-NVIDIA.txt")
    return {"source_model_sha256":sha256(source),"optimized_model_sha256":sha256(graph),"fused_regions":proofs}

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir",type=Path,required=True);parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--custom-op-library",type=Path);parser.add_argument("--speech-pcm",type=Path)
    parser.add_argument("--validation-frames",type=int,default=40);parser.add_argument("--validate-only",action="store_true")
    args=parser.parse_args()
    if args.validation_frames<2:parser.error("At least two recurrent hops are required")
    if not args.validate_only:print(json.dumps(optimize(args.source_dir,args.output_dir),indent=2))
    if args.custom_op_library:print(json.dumps(validate(args.source_dir,args.output_dir,args.custom_op_library,args.validation_frames,args.speech_pcm),indent=2))
    elif args.validate_only:parser.error("--validate-only requires --custom-op-library")
