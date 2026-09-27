"""Permalink-Sync (Query-Parameter <-> Session-State) und Presets."""
from dataclasses import dataclass
from typing import Any, Callable

import streamlit as st

import pc_constants as C


@dataclass(frozen=True)
class SettingSpec:
    key: str
    param: str
    default: Any
    cast: Callable[[str], Any]
    bounds: tuple | None = None  # (min, max) fuer numerische Regler, sonst None


def bounds(mode: str) -> dict:
    """Reglergrenzen fuer gap, abhaengig vom Modus (spread ist konstant)."""
    if mode == C.MODE_TRENNBAR:
        return dict(gap_min=C.GAP_TRENNBAR_MIN, gap_max=C.GAP_TRENNBAR_MAX,
                    gap_default=C.GAP_TRENNBAR_DEFAULT)
    return dict(gap_min=C.GAP_XOR_MIN, gap_max=C.GAP_XOR_MAX, gap_default=C.GAP_XOR_DEFAULT)


SETTING_SPECS = [
    SettingSpec("mode", "modus", C.MODE_TRENNBAR, str),
    SettingSpec("n", "n", C.N_DEFAULT, int, (C.N_MIN, C.N_MAX)),
    SettingSpec("seed", "seed", C.DEFAULT_SEED, int),
    SettingSpec("eta", "eta", C.ETA_DEFAULT, float, (C.ETA_MIN, C.ETA_MAX)),
    SettingSpec("max_epochs", "epochen", C.MAX_EPOCHS_DEFAULT, int,
                (C.MAX_EPOCHS_MIN, C.MAX_EPOCHS_MAX)),
    SettingSpec("gap", "gap", C.GAP_TRENNBAR_DEFAULT, float),
]


def init_session_state_defaults() -> None:
    for spec in SETTING_SPECS:
        if spec.key not in st.session_state:
            st.session_state[spec.key] = spec.default


def load_permalink_settings() -> None:
    params = st.query_params
    for spec in SETTING_SPECS:
        if spec.param in params and spec.key not in st.session_state:
            raw = params[spec.param]
            try:
                value = spec.cast(raw)
            except (TypeError, ValueError):
                continue
            if spec.bounds is not None:
                lo, hi = spec.bounds
                value = min(max(value, lo), hi)
            st.session_state[spec.key] = value
    if st.session_state.get("mode") not in C.MODES:
        st.session_state["mode"] = C.MODE_TRENNBAR


def sync_query_params(values: dict) -> None:
    for spec in SETTING_SPECS:
        if spec.key in values:
            st.query_params[spec.param] = str(values[spec.key])


def store_from_widget(key: str) -> None:
    st.session_state[key] = st.session_state[f"widget_{key}"]


def apply_preset(preset_key: str) -> None:
    preset = C.PRESETS[preset_key]
    for field in ("mode", "n", "gap", "eta", "max_epochs", "seed"):
        if field in preset:
            st.session_state[field] = preset[field]
            st.session_state[f"widget_{field}"] = preset[field]


def randomize_seed() -> None:
    import random
    new_seed = random.randint(0, 999_999)
    st.session_state["seed"] = new_seed
    st.session_state["widget_seed"] = new_seed
