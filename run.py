"""
Run simulated annealing
"""

# run example: python3 run.py

import sys
from timeit import default_timer

start = default_timer()

# my modules
import modules.mol as mol
import modules.wrap as wrap
import modules.read_input as read_input

# create class objects
m = mol.Xyz()
w = wrap.Wrapper()
p = read_input.Input_to_params("input.json")

# Optional command line args

run_id = sys.argv[1] if len(sys.argv) > 1 else 0
start_xyz_file = sys.argv[2] if len(sys.argv) > 2 else 0
target_file = sys.argv[3] if len(sys.argv) > 3 else 0
if len(sys.argv) > 4:
    p.tuning_ratio_target = float(sys.argv[4])

# Call the params function and add them to p object
p = w.run_xyz_openff_mm_params(p, start_xyz_file)

# Call the run function
w.run(p, run_id, start_xyz_file, target_file)

print("Total time: %3.2f s" % float(default_timer() - start))
