import os
import json
import sys

files = [os.path.join(sys.argv[1],f) for f in os.listdir(sys.argv[1]) if os.path.isfile(os.path.join(sys.argv[1],f))]

data = []
for f in files:
  with open(f, "r") as f_:
    data.append(json.load(f_))

with open(sys.argv[2], "w") as f:
  json.dump(data, f)
