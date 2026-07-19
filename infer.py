import argparse
import json

from selectground import SelectGround


parser = argparse.ArgumentParser(description="Ground one GUI instruction.")
parser.add_argument("--model", default="ruotian/SelectGround-8B")
parser.add_argument("--image", required=True)
parser.add_argument("--instruction", required=True)
parser.add_argument("--conground", action="store_true")
args = parser.parse_args()

grounder = SelectGround(args.model)
print(json.dumps(grounder.predict(args.image, args.instruction, conground=args.conground), indent=2))
