import netket as nk

# import h5py
import netket.experimental.logging.hdf5_log as h5_log
import json
import os
from collections.abc import Sequence


# def log_to_h5(log: nk.logging.AbstractLog, fname: str, write_mode="r+"):
#     f = h5py.File(fname, write_mode)
#     h5_log.tree_log(tree=log.data, root="Data", data=f)
#     f.close()


# def write_attributes(filename: str, group_name: str, write_mode="r+", **kwargs):
#     f = h5py.File(filename, write_mode)
#     group = f.create_group(group_name)
#     for key, val in kwargs.items():
#         group.attrs[f"{key}"] = val
#     f.close()


def write_attributes_json(filename: str, group_name: str, **kwargs):
    if os.path.exists(filename):
        updating_json = json.load(open(filename))
    else:
        updating_json = {}

    new_json = {}
    for key, val in kwargs.items():
        new_json[f"{group_name}/{key}"] = str(val)

    updating_json.update(new_json)
    json.dump(updating_json, open(filename, mode="w"))


def write_json(filename: str, gnames: Sequence, objs: Sequence):
    for gname, obj in zip(gnames, objs):
        write_attributes_json(filename, gname, **vars(obj))
