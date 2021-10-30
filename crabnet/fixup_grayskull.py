"""Touch up the conda recipe from grayskull using conda-souschef."""
import os
from os.path import join

from souschef.recipe import Recipe

import crabnet

os.system("grayskull pypi {0}=={1}".format(crabnet.__name__, crabnet.__version__))

fpath = join(crabnet.__name__, "meta.yaml")
fpath2 = join("scratch", "meta.yaml")
my_recipe = Recipe(load_file=fpath)
my_recipe["requirements"]["host"].append("flit")
my_recipe["requirements"]["run"].append("pytorch >=1.9.0")
my_recipe["requirements"]["run"].append("cudatoolkit <11.4")
my_recipe.save(fpath)
my_recipe.save(fpath2)
