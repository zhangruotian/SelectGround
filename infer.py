import argparse
import json

from selectground import SelectGround


parser = argparse.ArgumentParser(description="Ground one GUI instruction.")
parser.add_argument("--model", default="ruotian/SelectGround-8B")
parser.add_argument("--image", required=True)
parser.add_argument("--instruction", required=True)
parser.add_argument("--lcr", action="store_true")
parser.add_argument("--benchmark", choices=("screenspot_pro", "mmbench_gui_l2", "osworld_g"), default="screenspot_pro")
args = parser.parse_args()

grounder = SelectGround(args.model)
print(json.dumps(grounder.predict(args.image, args.instruction, lcr=args.lcr, benchmark=args.benchmark), indent=2))
