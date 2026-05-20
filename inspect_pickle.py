import pickle
import pprint
import sys

path = sys.argv[1]

with open(path, "rb") as f:
    obj = pickle.load(f)

print("TYPE:", type(obj))

pprint.pprint(obj.__dict__)

# path: experiments/crop-0.2-0.25/options-and-config.pickle
# python -i inspect_pickle.py "experiments/crop-0.2-0.25/options-and-config.pickle"