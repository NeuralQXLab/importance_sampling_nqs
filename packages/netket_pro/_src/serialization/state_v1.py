import zipfile
import time
import socket
import sys

import orjson
import json

from pathlib import Path

import jax

from contextlib import nullcontext

from netket.utils.version_check import module_version

from netket_pro import distributed
from nqxpack._src.lib_v1 import (
    serialize_object,
    deserialize_object,
    resolution,
)
from nqxpack._src.lib_v1 import asset_lib
from nqxpack._src.contextmgr import PackingContext

_CONFIG_FILENAME = "config.json"
_METADATA_FILENAME = "metadata.json"
_STATE_DICT_FILENAME = "state.msgpack"
_VARS_FNAME = "model.weights"  # Will become e.g. "model.weights.h5"

_FORMAT_VERSION = 1.1


def get_metadata():
    return {
        "format": "NetKet",
        "format_version": _FORMAT_VERSION,
        "python_version": sys.version,
        "netket_version": module_version("netket"),
        "flax_version": module_version("flax"),
        "time": time.time(),
        "hostname": socket.gethostname(),
        "n_devices": jax.device_count(),
    }


def get_config():
    pass


def save_mcstate_v1(state, fileobj):
    if not isinstance(fileobj, Path):
        fileobj = Path(fileobj)

    # Check that extension is .nk
    if fileobj.suffix != ".nk":
        fileobj = fileobj.with_suffix(".nk")

    metadata_json = get_metadata()

    fopen = lambda: (
        zipfile.ZipFile(fileobj, "w")
        if distributed.is_master_process()
        else nullcontext()
    )

    orjson_options = orjson.OPT_APPEND_NEWLINE | orjson.OPT_INDENT_2
    with fopen() as zf:
        with PackingContext(
            asset_manager=asset_lib.ArchiveAssetManager(zf, root="state/")
        ):
            config_json = serialize_object({"state": state})

            if distributed.is_master_process():
                with zf.open(_CONFIG_FILENAME, "w") as f:
                    f.write(orjson.dumps(config_json, option=orjson_options))

                metadata_json["package_versions"] = resolution.PACKAGE_VERSIONS
                with zf.open(_METADATA_FILENAME, "w") as f:
                    f.write(orjson.dumps(metadata_json, option=orjson_options))


def load_mcstate_v1(cls, fileobj):
    with zipfile.ZipFile(fileobj, "r") as zf:
        metadata = zf.read(_METADATA_FILENAME)
        metadata = json.loads(metadata)

        # validate
        if metadata["format"] != "NetKet":
            raise ValueError("Invalid file format.")
        if metadata["format_version"] > _FORMAT_VERSION:
            raise ValueError(
                f"""
                             File was saved with a more recent version of netket_pro than your current installation:

                                File version     : {metadata['format_version']}
                                Supported version: {_FORMAT_VERSION}

                             You should update NetKet pro.
                             """
            )

        config = zf.read(_CONFIG_FILENAME)
        with PackingContext(
            asset_manager=asset_lib.ArchiveAssetManager(zf, root="state/")
        ):

            state_obj_dict = json.loads(config)["state"]

            # Fix a but that appeared briefly
            # TODO: Put this somewhere reasoanbly
            if "_target_" in state_obj_dict:
                if state_obj_dict["_target_"] == "netket.vqs.mc.mc_state.state.MCState":
                    state_obj_dict["_target_"] = "#netket.vqs.mc.mc_state.state.MCState"

            state = deserialize_object(state_obj_dict)

    return state
