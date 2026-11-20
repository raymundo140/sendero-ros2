"""The shipped parameter file documents every parameter with its default."""
# Author: Raymundo Gomez

import dataclasses
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sendero.config import (PROFILE_FIELDS, NavigationConfig,  # noqa: E402
                            build_profiles)

yaml = pytest.importorskip('yaml')
CONFIG = os.path.join(os.path.dirname(__file__), '..', 'config')
PROFILE_KEYS = {'profile_names', 'default_profile', 'profiles'}


def _load(name):
    with open(os.path.join(CONFIG, name)) as f:
        return yaml.safe_load(f)['/**']['ros__parameters']


def test_default_file_lists_every_parameter_with_its_default():
    params = _load('sendero.yaml')
    fields = {f.name for f in dataclasses.fields(NavigationConfig)}
    unknown = set(params) - fields - PROFILE_KEYS
    assert not unknown, f'unknown keys in sendero.yaml: {unknown}'
    missing = fields - set(params)
    assert not missing, f'parameters not documented in sendero.yaml: {missing}'
    default = NavigationConfig()
    for key, value in params.items():
        if key in PROFILE_KEYS:
            continue
        assert value == getattr(default, key), key


@pytest.mark.parametrize('name', sorted(n for n in os.listdir(CONFIG)
                                        if n.endswith('.yaml')))
def test_every_shipped_file_builds(name):
    params = _load(name)
    params.pop('use_sim_time', None)
    profiles = params.pop('profiles', {}) or {}
    for prof in profiles.values():
        assert set(prof) <= set(PROFILE_FIELDS)
    cfg = NavigationConfig(**params)
    cfg.validate()
    built = build_profiles(cfg, profiles)
    assert set(built) == set(cfg.profile_names)
