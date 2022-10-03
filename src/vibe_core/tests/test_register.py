from vibe_core.data import data_registry, DataVibe


def test_register_type():
    class InternalFakeType(DataVibe):
        pass

    assert data_registry.retrieve("InternalFakeType") == InternalFakeType
