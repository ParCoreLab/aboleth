import copy
import unittest
from aboleth.amd_topology import registered_amd_peer_link
from aboleth.endpoint_profiles import AMD_MI355X_VF_XGMI_EP2
from aboleth.planning import EndpointPartitionFacts,plan_endpoint_work,plan_transport_partitions


def images():
    nodes={}
    for own,peer in [('8','9'),('9','8')]:
        nodes[own]=dict(properties=dict(hive_id=123),io_links=[dict(node_to=int(peer),
            type=11,flags=1,weight=15,min_bandwidth=76000,max_bandwidth=76000)])
    return [dict(pci='0000:b3:00.0',node='8',platform='fixture',kfd=copy.deepcopy(nodes)),
            dict(pci='0000:bb:00.0',node='9',platform='fixture',kfd=copy.deepcopy(nodes))]


class TopologyTests(unittest.TestCase):
    def test_direct_symmetric_xgmi(self):
        self.assertEqual(registered_amd_peer_link(images()),AMD_MI355X_VF_XGMI_EP2.peer_link)

    def test_unqualified_world_sizes(self):
        self.assertEqual(registered_amd_peer_link(images()[:1]),'')
        self.assertEqual(registered_amd_peer_link(images()*2),'')

    def test_missing_identity_or_topology(self):
        self.assertEqual(registered_amd_peer_link([None,None]),'')
        image=images();del image[0]['kfd']['9']
        self.assertEqual(registered_amd_peer_link(image),'')

    def test_duplicate_pci(self):
        image=images();image[1]['pci']=image[0]['pci']
        self.assertEqual(registered_amd_peer_link(image),'')

    def test_peer_platform_mismatch(self):
        image=images();image[1]['platform']='other'
        self.assertEqual(registered_amd_peer_link(image),'')

    def test_disabled_noncoherent_or_no_atomic_links(self):
        for flag in [0,3,5,9,17]:
            image=images();image[0]['kfd']['8']['io_links'][0]['flags']=flag
            self.assertEqual(registered_amd_peer_link(image),'')

    def test_asymmetric_or_indirect_link(self):
        for field,value in [('weight',30),('type',2),('max_bandwidth',38000)]:
            image=images();image[0]['kfd']['8']['io_links'][0][field]=value
            self.assertEqual(registered_amd_peer_link(image),'')

    def test_different_hive(self):
        image=images();image[0]['kfd']['9']['properties']['hive_id']=124
        self.assertEqual(registered_amd_peer_link(image),'')

    def test_cpp_profile_and_unknown_participants(self):
        profile=AMD_MI355X_VF_XGMI_EP2
        partition=plan_transport_partitions(((1,'p2p'),))[0]
        def plan(platforms,link):
            return plan_endpoint_work(name='amd-profile-test',payload_bytes=131072,max_items=32,
                partitions=(EndpointPartitionFacts(partition,'none',profile.platform,16,
                    tuple((p,'none',16) for p in platforms),link,(link,link)),),
                profiles=(profile,),rank_count=2,grouped_receiver_supported=True)[0]
        selected=plan([profile.platform,profile.platform],profile.peer_link)
        self.assertEqual((selected.lanes,selected.sender_warps),(8,8))
        self.assertEqual(selected.profile_identity,profile.identity)
        for platforms,link in [([profile.platform,'unknown'],profile.peer_link),
                               ([profile.platform,profile.platform],'')]:
            fallback=plan(platforms,link)
            self.assertEqual((fallback.lanes,fallback.sender_warps,fallback.profile_identity),(1,1,''))


if __name__=='__main__':unittest.main()
