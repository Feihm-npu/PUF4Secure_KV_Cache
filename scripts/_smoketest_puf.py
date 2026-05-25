"""Quick sanity tests for puf_sim and puf_basis."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from puf4secure_kvcache.puf_sim import make_puf
from puf4secure_kvcache.puf_basis import (
    make_orthogonal, protect_kv_cache, recover_kv_cache, relative_l2_error,
    ProtectionSpec, LayoutSpec,
)


def test_orth():
    for kind in ["signed_perm", "givens", "hadamard", "householder", "qr"]:
        O = make_orthogonal(kind, 128, seed=42, device=torch.device("cpu"))
        I = O @ O.T
        err = (I - torch.eye(128)).norm().item()
        print(f"{kind:14s} ||O O^T - I|| = {err:.3e}")


def test_puf_determinism():
    p = make_puf("device_A")
    a = p.derive_seed(layer=3, group=2, block=0, purpose="V")
    b = p.derive_seed(layer=3, group=2, block=0, purpose="V")
    c = p.derive_seed(layer=3, group=2, block=0, purpose="K")
    print(f"same-seed equal: {a == b}, different purpose: {a != c}")
    pB = make_puf("device_B")
    d = pB.derive_seed(layer=3, group=2, block=0, purpose="V")
    print(f"cross-device differs: {a != d}")


def test_protect_recover_roundtrip():
    # Synthetic KV cache: 2 layers, kv_h=8, seq=18, head_dim=128
    torch.manual_seed(0)
    kv = [(torch.randn(1, 8, 18, 128, dtype=torch.bfloat16),
           torch.randn(1, 8, 18, 128, dtype=torch.bfloat16)) for _ in range(2)]
    puf_A = make_puf("device_A")
    puf_B = make_puf("device_B")
    for kind in ["signed_perm", "givens", "hadamard", "householder", "qr"]:
        for layout in ["none", "row", "block"]:
            spec = ProtectionSpec(kind=kind, layout=LayoutSpec(kind=layout, block_size=4))
            prot = protect_kv_cache(kv, puf_A, spec)
            rec = recover_kv_cache(prot, puf_A, spec)
            wrong = recover_kv_cache(prot, puf_B, spec)
            err = relative_l2_error(kv, rec)
            werr = relative_l2_error(kv, wrong)
            print(f"{kind:14s} layout={layout:5s}  same_dev V={err['V_rel_l2']:.2e} K={err['K_rel_l2']:.2e} | wrong V={werr['V_rel_l2']:.2e}")


if __name__ == "__main__":
    test_orth()
    print()
    test_puf_determinism()
    print()
    test_protect_recover_roundtrip()
