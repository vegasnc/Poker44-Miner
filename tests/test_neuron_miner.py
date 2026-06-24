from __future__ import annotations


def test_neuron_miner_entrypoint_imports():
    from neurons.miner import Miner, POKER44_RUNTIME_AVAILABLE

    assert Miner is not None
    assert isinstance(POKER44_RUNTIME_AVAILABLE, bool)
