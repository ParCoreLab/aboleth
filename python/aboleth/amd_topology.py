"""Read HIP/KFD facts for hardware-scoped P2P endpoint profiles."""
from pathlib import Path
import ctypes, hashlib, json
import torch


def properties(path):
    return {key: int(value, 0) for key, value in
            (line.split() for line in Path(path).read_text().splitlines())}


def kfd_image(root='/sys/class/kfd/kfd/topology/nodes'):
    nodes={}
    for p in sorted(Path(root).iterdir(),key=lambda x:int(x.name)):
        facts=properties(p/'properties')
        if not int((p/'gpu_id').read_text()):continue
        nodes[p.name]=dict(properties=facts,io_links=[properties(q/'properties')
            for q in sorted((p/'io_links').iterdir(),key=lambda x:int(x.name))])
    return nodes


def device_image(device):
    hip=ctypes.CDLL('libamdhip64.so.7')
    hip.hipDeviceGetPCIBusId.argtypes=[ctypes.c_char_p,ctypes.c_int,ctypes.c_int]
    hip.hipDeviceGetPCIBusId.restype=ctypes.c_int
    buf=ctypes.create_string_buffer(32)
    assert hip.hipDeviceGetPCIBusId(buf,len(buf),device)==0
    pci=buf.value.decode().lower()
    domain,bus,devfunc=pci.split(':');dev,func=devfunc.split('.')
    domain=int(domain,16);location=(int(bus,16)<<8)|(int(dev,16)<<3)|int(func,16)
    image=kfd_image();matches=[(n,v['properties']) for n,v in image.items()
        if v['properties']['domain']==domain and v['properties']['location_id']==location]
    assert len(matches)==1,(pci,matches)
    node,facts=matches[0];props=torch.cuda.get_device_properties(device)
    hardware={k:facts[k] for k in ['vendor_id','device_id','gfx_target_version','wave_front_size','num_xcc','fw_version']}
    platform=json.dumps(['amd-kfd-p2p-v1',props.name,props.gcnArchName,torch.version.hip,Path('/sys/module/amdgpu/version').read_text().strip(),hardware],sort_keys=True,separators=(',',':'))
    return dict(pci=pci,node=node,uuid=str(props.uuid),hardware=hardware,platform=platform,kfd=image)


def common_link(images):
    assert len(images)>=2 and len({x['pci'] for x in images})==len(images)
    assert len({x['platform'] for x in images})==1
    nodes=[x['node'] for x in images];records=[]
    for image in images:
        table=image['kfd'];own=image['node'];hive=table[own]['properties']['hive_id']
        assert hive!=0
        for peer in nodes:
            if peer==own:continue
            assert table[peer]['properties']['hive_id']==hive
            links=[v for v in table[own]['io_links'] if v['node_to']==int(peer)]
            assert len(links)==1
            link=links[0]
            # Linux KFD UAPI: type11 is XGMI; bit0 enabled, bits1-4 limitations.
            assert link['type']==11 and link['flags']==1
            assert link['weight']==15 and link['min_bandwidth']==link['max_bandwidth']>0
            records.append({k:link[k] for k in ['type','weight','min_bandwidth','max_bandwidth','flags']})
    assert all(r==records[0] for r in records)
    return json.dumps(['amd-kfd-direct-link-v1',records[0]],sort_keys=True,separators=(',',':'))


def registered_amd_device(device):
    """Return registration facts, or leave unsupported/missing topology unqualified."""
    try:
        return device_image(device)
    except (OSError, RuntimeError, ValueError, KeyError, AttributeError, AssertionError):
        return None


def registered_amd_peer_link(images):
    """EP2 is the only calibrated scope; other world sizes retain generic geometry."""
    if len(images)!=2 or any(image is None for image in images):
        return ""
    try:
        return common_link(images)
    except (OSError, RuntimeError, ValueError, KeyError, TypeError, AssertionError):
        return ""
