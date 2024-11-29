from vibe_core.data import DataVibe, data_registry


def test_register_type():
    class InternalFakeType(DataVibe):
        pass

    assert data_registry.retrieve("InternalFakeType") == InternalFakeType
