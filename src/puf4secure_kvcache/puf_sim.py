"""Simulated PUF root for the PUF-basis defense PoC.

A real silicon PUF gives noisy device-unique bits which are fed through a fuzzy
extractor (helper-data assisted) to yield a stable secret key per device.  In
software we model the same interface:

    derive_seed(device_id, session_nonce, layer, group, block, purpose) -> int
    derive_bytes(device_id, session_nonce, layer, group, block, purpose) -> bytes

The simulator supports three modes:

  stable        : deterministic regeneration, no noise.
  noisy         : raw response gets bit-flips at a given BER.  A fuzzy-extractor
                  stub corrects up to `correction_capacity` flipped bits and
                  recovers the stable root; beyond that, derivation falls
                  through to a different (corrupted) key.
  wrong_device  : a different device root entirely, used for cross-device
                  non-migratability tests.

The "device root" itself is a fixed 32-byte secret indexed by `device_id`.
Per-call seeds are derived via HMAC-SHA256 over the structured context so that
different (layer, group, block, purpose) coordinates produce uncorrelated
seeds even within a single session.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import struct
from dataclasses import dataclass


# Master table of simulated device roots.  In a real deployment this is the
# silicon-bound PUF response after fuzzy extraction; here it is a deterministic
# pseudorandom 32-byte string keyed by device_id.
_MASTER = b"PUF4Secure_KVcache::device_root_v1"


def _device_root(device_id: str) -> bytes:
    return hashlib.sha256(_MASTER + b"|" + device_id.encode("utf-8")).digest()


def _xor(a: bytes, b: bytes) -> bytes:
    return bytes(x ^ y for x, y in zip(a, b))


def _flip_bits(buf: bytes, ber: float, rng: "os.urandom") -> bytes:
    """Flip each bit independently with probability `ber`."""
    if ber <= 0.0:
        return buf
    out = bytearray(buf)
    n_bits = len(out) * 8
    # Sample flips deterministically from the supplied RNG callable.
    flips = rng(n_bits)
    for i in range(n_bits):
        if flips[i] < ber * 256:
            out[i >> 3] ^= 1 << (i & 7)
    return bytes(out)


@dataclass
class PUFSim:
    """Simulated PUF oracle bound to a single device_id and session_nonce.

    mode:
      "stable"        - perfect regeneration of device_id's root
      "noisy"         - simulate raw bit errors at `ber`, then fuzzy-correct
                         if hamming distance <= correction_capacity
      "wrong_device"  - claim device_id but actually return a different root
    """

    device_id: str
    session_nonce: bytes = b"\x00" * 16
    mode: str = "stable"
    ber: float = 0.0
    correction_capacity: int = 64           # bits the fuzzy extractor can fix
    impostor_device_id: str | None = None    # used by mode == "wrong_device"
    noise_seed: int = 0                      # RNG seed for reproducible noise

    def _reconstructed_root(self) -> bytes:
        true_root = _device_root(self.device_id)
        if self.mode == "stable":
            return true_root
        if self.mode == "wrong_device":
            impostor = self.impostor_device_id or (self.device_id + "_attacker")
            return _device_root(impostor)
        if self.mode == "noisy":
            # Simulate raw PUF response with bit flips, then check whether the
            # fuzzy extractor can recover the original codeword.
            seed_block = hashlib.sha256(
                b"puf_noise|" + self.device_id.encode("utf-8")
                + b"|" + self.session_nonce
                + struct.pack(">q", self.noise_seed)
            ).digest()
            # Expand to n_bits bytes (one byte of randomness per bit decision).
            n_bits_needed = len(true_root) * 8
            blocks = []
            ctr = 0
            while sum(len(b) for b in blocks) < n_bits_needed:
                blocks.append(hashlib.sha256(seed_block + struct.pack(">I", ctr)).digest())
                ctr += 1
            noise_buf = b"".join(blocks)
            flipped = _flip_bits(true_root, self.ber, lambda n: noise_buf[:n])
            hamming = sum(bin(a ^ b).count("1") for a, b in zip(flipped, true_root))
            if hamming <= self.correction_capacity:
                # Fuzzy extractor corrects the errors -> stable root
                return true_root
            # Beyond correction capacity: extractor outputs garbled key
            return hashlib.sha256(b"corrupt|" + flipped).digest()
        raise ValueError(f"Unknown PUFSim mode: {self.mode!r}")

    def derive_bytes(
        self,
        *,
        layer: int = -1,
        group: int = -1,
        block: int = -1,
        purpose: str = "",
        n_bytes: int = 32,
    ) -> bytes:
        """HMAC-SHA256(root, structured_context) -> n_bytes."""
        root = self._reconstructed_root()
        ctx = (
            self.session_nonce
            + struct.pack(">qqqq", layer, group, block, len(purpose))
            + purpose.encode("utf-8")
        )
        out = b""
        counter = 0
        while len(out) < n_bytes:
            out += hmac.new(root, ctx + struct.pack(">I", counter), hashlib.sha256).digest()
            counter += 1
        return out[:n_bytes]

    def derive_seed(self, **kwargs) -> int:
        """Derive a 64-bit unsigned int suitable for torch.Generator.manual_seed."""
        raw = self.derive_bytes(n_bytes=8, **kwargs)
        return int.from_bytes(raw, "big", signed=False) & ((1 << 63) - 1)


def make_puf(
    device_id: str,
    *,
    session_nonce: bytes | None = None,
    mode: str = "stable",
    **kwargs,
) -> PUFSim:
    if session_nonce is None:
        session_nonce = hashlib.sha256(b"session|" + device_id.encode("utf-8")).digest()[:16]
    return PUFSim(device_id=device_id, session_nonce=session_nonce, mode=mode, **kwargs)
