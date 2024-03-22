# %matplotlib inline
import logging

from icesat2waves.local_modules import m_colormanager_ph3 as M_color

_logger = logging.getLogger(__name__)

mconfig["paths"]
path = mconfig["paths"]["config"]
A = M_color.ase_to_json(path + "color_def.ase")

B = dict()
for i in A[0]["swatches"]:
    B[i["name"]] = i["data"]["values"]
    _logger.debug("swatch name: %s, values: %s", i["name"], i["data"]["values"])

rels = dict()


rels["plus"] = B["red"]
rels["minus"] = B["blue"]

rels["blue"] = B["blue"]
rels["lightblue"] = B["cascade3"]
rels["darkblue"] = B["cascade1"]


rels["white"] = B["white"]
rels["gridcolor"] = B["gridcolor"]
rels["grey"] = B["gray"]

rels["orange"] = B["orange"]
rels["red"] = B["red"]
rels["green"] = B["green"]

rels["cascade1"] = B["cascade1"]
rels["cascade2"] = B["cascade2"]
rels["cascade3"] = B["cascade3"]
rels["cascade4"] = B["gridcolor"]

rels["rascade1"] = B["rascade2"]
rels["rascade2"] = B["rascade1"]
rels["rascade3"] = B["rascade3"]

rels["aug1"] = B["orange"]
rels["aug2"] = B["green"]

rels["gt1l"] = B["rascade1"]
rels["gt1r"] = B["rascade3"]

rels["gt2l"] = B["green1"]
rels["gt2r"] = B["green2"]

rels["gt3l"] = B["cascade1"]
rels["gt3r"] = B["blue"]

rels["group1"] = B["rascade1"]
rels["group2"] = B["green1"]
rels["group3"] = B["cascade1"]

B["rels"] = rels

M_color.json_save("color_def", path, B)
