import pytest

from vsdenoise import MVToolsPreset


def test_presets_properties() -> None:
    preset_sad = MVToolsPreset.HQ_SAD
    assert preset_sad is not None
    assert preset_sad.analyze_args["blksize"] == 16

    preset_coh = MVToolsPreset.HQ_COHERENCE
    assert preset_coh is not None
    assert preset_coh.recalculate_args.get("satd") is True


def test_preset_mapping_methods() -> None:
    preset = MVToolsPreset.HQ_SAD

    # __str__
    assert str(preset) == str(preset._dict)

    # __getitem__
    assert preset["analyze_args"] == preset._dict["analyze_args"]
    with pytest.raises(KeyError):
        preset["non_existent_key"]

    # __iter__
    assert set(iter(preset)) == set(iter(preset._dict))

    # __len__
    assert len(preset) == len(preset._dict)

    # __or__
    merged = preset | {"custom_key": "custom_value"}
    assert isinstance(merged, dict)
    assert merged["custom_key"] == "custom_value"
    assert merged["analyze_args"] == preset["analyze_args"]

    # __ror__
    rmerged = {"custom_key": "custom_value"} | preset
    assert isinstance(rmerged, dict)
    assert rmerged["custom_key"] == "custom_value"
    assert rmerged["analyze_args"] == preset["analyze_args"]

    # copy
    copied = preset.copy()
    assert isinstance(copied, dict)
    assert copied == preset._dict


def test_preset_getattr() -> None:
    preset = MVToolsPreset.HQ_SAD

    # Defined in annotations and present in _dict
    assert preset.analyze_args == preset._dict["analyze_args"]

    # Defined in annotations but NOT present in _dict
    with pytest.raises(AttributeError):
        preset.pel

    # NOT defined in annotations and NOT present in __dict__
    with pytest.raises(AttributeError):
        preset.non_existent_attr
