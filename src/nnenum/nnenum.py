'''
nnenum vnnlib front end

usage: "python3 nnenum.py -o <onnx_file> -v <vnnlib_file> [-t timeout=None] [-f outfile=None] [-p processes] [-s settings=auto]"

Ali A. Bigdeli
June 2022
'''


import sys
import argparse

import numpy as np

from pathlib import Path

from nnenum.enumerate import enumerate_network
from nnenum.settings import Settings
from nnenum.result import Result
from nnenum.onnx_network import load_onnx_network_optimized, load_onnx_network
from nnenum.specification import Specification, DisjunctiveSpec
from nnenum.vnnlib import get_num_inputs_outputs, read_vnnlib_simple, read_io_vnnlib
import nnenum.setting_cat as setting_cat
import time
import re
import onnxruntime as ort

def make_spec(vnnlib_filename, onnx_filename):
    '''make Specification

    returns a pair: (list of [box, Specification], inp_dtype)
    '''

    num_inputs, num_outputs, inp_dtype = get_num_inputs_outputs(onnx_filename)
    vnnlib_spec = read_vnnlib_simple(vnnlib_filename, num_inputs, num_outputs)

    rv = []

    for box, spec_list in vnnlib_spec:
        if len(spec_list) == 1:
            mat, rhs = spec_list[0]
            spec = Specification(mat, rhs)
        else:
            spec_obj_list = [Specification(mat, rhs) for mat, rhs in spec_list]
            spec = DisjunctiveSpec(spec_obj_list)

        rv.append((box, spec))

    return rv, inp_dtype

def run_io_verify(vnnlib_filename, onnx_filename, timeout=None, outfile=None):
    '''verify 'LinearizeNN_benchmark2024'

    returns a pair: (list of [box, Specification], inp_dtype)
    '''

    num_inputs, num_outputs, inp_dtype = get_num_inputs_outputs(onnx_filename)
    init_box, normalize_models_slope, models_intercept, model_range = read_io_vnnlib(vnnlib_filename, num_inputs, num_outputs)

    start_time = time.time()
    set_exact_settings()
    Settings.RESULT_SAVE_STARS = True
    if timeout is not None: Settings.TIMEOUT = timeout
    spec = None
    # network = load_onnx_network(onnx_filename)
    network = load_onnx_network_optimized(onnx_filename)
    res = enumerate_network(init_box, network, spec)
    star_list = res.stars
    time_taken = time.time() - start_time
    # print(f"######## Time taken for getting final stars: {time_taken} seconds")

    max_total = -np.inf
    min_total = np.inf
    maxlist = []
    minlist = []
    maxpoints = []
    minpoints = []
    result_str = 'none' # gets overridden
    cinput = None # counterexample input
    coutput = None # counterexample output
    retrun_ce_based_on_inputs = True
    try:
        start_time = time.time()
        for idx, star in enumerate(star_list):
            bias =  star.bias - models_intercept
            direction_vec = np.array([star.a_mat[0, 0], star.a_mat[0, 1], star.a_mat[0, 2] - normalize_models_slope[0], star.a_mat[0, 3] - normalize_models_slope[1]])
            min_point = star.lpi.minimize(direction_vec)
            min_value = min_point @ direction_vec + bias
            max_point = star.lpi.minimize(-1 * direction_vec)
            max_value = max_point @ direction_vec + bias

            assert min_value <= max_value

            minlist.append(min_value)
            maxlist.append(max_value)
            minpoints.append(min_point)
            maxpoints.append(max_point)
            if min_value < min_total:
                min_total = min_value
            if max_value > max_total:
                max_total = max_value
        time_taken = time.time() - start_time
        # print(f"######## Time taken for finding min/max values: {time_taken} seconds")
        if model_range[0] <= min_total and max_total <= model_range[1]:
            result_str = "unsat"
            print("network is SAFE")
        else:
            result_str = "sat"
            print("network is UNSAFE") 
            if model_range[0] > min_total:
                cinput = minpoints[np.argmin(minlist)]
                coutput = min_total
            else:
                cinput = maxpoints[np.argmax(maxlist)]
                coutput = max_total
            if retrun_ce_based_on_inputs:
                ### manual calulation:
                # y_temp_list = [0, 0]
                # y_temp_list[0] = coutput.item() - normalize_models_slope[0] * cinput[2] - normalize_models_slope[1] * cinput[3]
                # y_temp_list[1] = - coutput.item() + normalize_models_slope[0] * cinput[2] + normalize_models_slope[1] * cinput[3]
                # coutput = np.array(y_temp_list)
                
                ### using onnx model:
                match = re.search(r'prop_(\d+)_(\d+)', vnnlib_filename)
                if match:
                    num1 = int(match.group(1))
                    num2 = int(match.group(2))
                onnx_run_name = onnx_filename.replace(".onnx", f"_{num1}_{num2}.onnx")
                
                # Load the ONNX model and run inference
                try:
                    sess = ort.InferenceSession(onnx_run_name)
                    input_names = [inp.name for inp in sess.get_inputs()]
                    
                    # Prepare input - ensure it's the right shape and type
                    cinput_array = np.array(cinput, dtype=np.float32).reshape(1, -1)
                    inputs = {input_names[0]: cinput_array}
                    
                    # Run inference
                    outputs = sess.run(None, inputs)
                    coutput = outputs[0].flatten()  # Get the output and flatten to 1D
                    
                except Exception as e:
                    print(f"Error running ONNX inference on {onnx_run_name}: {e}")
                    # Fall back to the original coutput if ONNX inference fails
            

    except Exception as e:
        # Code to execute if any other type of error occurs
        result_str = 'error'
        print(f"An error occurred: {e}")

    if outfile is not None:
        with open(outfile, 'w', encoding="utf-8") as f:
            f.write(result_str)

            if result_str == "sat":
                # print counterexamples
                
                for i, x in enumerate(cinput):
                    if i == 0:
                        f.write('\n(')
                    else:
                        f.write('\n')
                    
                    f.write(f"(X_{i} {x})")

                ###########

                for i, y in enumerate(coutput):
                    f.write(f"\n(Y_{i} {y})")

                f.write(')')

def set_control_settings():
    'set settings for smaller control benchmarks'

    Settings.TIMING_STATS = False
    Settings.PARALLEL_ROOT_LP = False
    Settings.SPLIT_IF_IDLE = False
    Settings.PRINT_OVERAPPROX_OUTPUT = False
    Settings.TRY_QUICK_OVERAPPROX = True

    Settings.CONTRACT_ZONOTOPE_LP = True
    Settings.CONTRACT_LP_OPTIMIZED = True
    Settings.CONTRACT_LP_TRACK_WITNESSES = True

    Settings.OVERAPPROX_BOTH_BOUNDS = False

    Settings.BRANCH_MODE = Settings.BRANCH_OVERAPPROX
    Settings.OVERAPPROX_GEN_LIMIT_MULTIPLIER = 1.5
    Settings.OVERAPPROX_LP_TIMEOUT = 0.02
    Settings.OVERAPPROX_MIN_GEN_LIMIT = 70

def set_exact_settings():
    'set settings for smaller control benchmarks'

    #Settings.TIMING_STATS = True
    Settings.TRY_QUICK_OVERAPPROX = False

    Settings.CONTRACT_ZONOTOPE_LP = True
    Settings.CONTRACT_LP_OPTIMIZED = True
    Settings.CONTRACT_LP_TRACK_WITNESSES = True

    Settings.OVERAPPROX_BOTH_BOUNDS = False

    Settings.BRANCH_MODE = Settings.BRANCH_EXACT

def set_image_settings():
    'set settings for larger image benchmarks'

    Settings.COMPRESS_INIT_BOX = True
    Settings.BRANCH_MODE = Settings.BRANCH_OVERAPPROX
    Settings.TRY_QUICK_OVERAPPROX = False
    
    Settings.OVERAPPROX_MIN_GEN_LIMIT = np.inf
    Settings.SPLIT_IF_IDLE = False
    Settings.OVERAPPROX_LP_TIMEOUT = np.inf
    #Settings.TIMING_STATS = True

    # contraction doesn't help in high dimensions
    #Settings.OVERAPPROX_CONTRACT_ZONO_LP = False
    Settings.CONTRACT_ZONOTOPE = False
    Settings.CONTRACT_ZONOTOPE_LP = False

def main():
    'main entry point'

    parser = argparse.ArgumentParser() # description is optional
    parser.add_argument('-o', '--onnx', type=str, required=True, help="relative path to onnx file")
    parser.add_argument('-v', '--vnnlib', type=str, required=True, help="relative path to vnnlib file")
    parser.add_argument('-t', '--timeout', default=None, help="timeout in seconds. default is no timeout")
    parser.add_argument('-f', '--outfile', type=str, default=None, help="filename the result save on, e.g.: out.txt, out.csv")
    parser.add_argument('-p', '--processes', type=int, help="number of processes to use")
    parser.add_argument('-s', '--settings', type=str, default="auto", help="settings to running the tool, it can be 'control' or 'image' or name of the dataset for more specified setting. default is auto")
    args = parser.parse_args()
    
    if args.onnx:
        onnx_filename = args.onnx
    
    if args.vnnlib:
        vnnlib_filename = args.vnnlib
    
    timeout = int(float(args.timeout)) if args.timeout else None
    
    outfile = args.outfile
    
    if args.processes:
        processes = args.processes
        Settings.NUM_PROCESSES = processes
    
    if args.settings:
        settings_str = args.settings
        settings_str = re.sub(r"^(202[0-9]_|202[0-9])", "", settings_str)
        settings_str = re.sub(r"(_202[0-9]|202[0-9])$", "", settings_str)
    else:
        settings_str = "auto"
        
    
    # if len(sys.argv) < 3:
    #     print('usage: "python3 nnenum.py <onnx_file> <vnnlib_file> [timeout=None] [outfile=None] [processes=<auto>]"')
    #     sys.exit(1)

    # onnx_filename = sys.argv[1]
    # vnnlib_filename = sys.argv[2]
    # timeout = None
    # outfile = None

    # if len(sys.argv) >= 4:
    #     timeout = float(sys.argv[3])

    # if len(sys.argv) >= 5:
    #     outfile = sys.argv[4]

    # if len(sys.argv) >= 6:
    #     processes = int(sys.argv[5])
    #     Settings.NUM_PROCESSES = processes

    # if len(sys.argv) >= 7:
    #     settings_str = sys.argv[6]
    # else:
    #     settings_str = "auto"

    ####################################
    ####################################
    ####################################
    # VNN-COMP 2022 SPECIFIC
    
    if Path(onnx_filename + ".converted").is_file():
        onnx_filename = onnx_filename + ".converted"
        print(f"NOTE: Using converted onnx path: {onnx_filename}")
        
    ####################################
    ####################################
    ####################################

    #
    if settings_str == "linearizenn" and ("test" not in onnx_filename.split('/')[-1]):
        onnx_filename = re.sub(r"_\d+_\d+", "", onnx_filename)
        vnnlib_filename = vnnlib_filename.replace(".vnnlib", "_io.vnnlib")
        run_io_verify(vnnlib_filename, onnx_filename, timeout, outfile)
        return
    else:
        spec_list, input_dtype = make_spec(vnnlib_filename, onnx_filename)

    try:
        network = load_onnx_network_optimized(onnx_filename)
    except:
        # cannot do optimized load due to unsupported layers
        network = load_onnx_network(onnx_filename)

    result_str = 'none' # gets overridden

    num_inputs = len(spec_list[0][0])

    if settings_str == "auto":
        if num_inputs < 700:
            set_control_settings()
        else:
            set_image_settings()
    elif "cora" in settings_str:
        set_image_settings()
        Settings.LP_SOLVER = "Gurobi"
    elif "safenlp" in settings_str:
        set_image_settings()
    elif settings_str in ["cifar2020", "cifar"]:
        set_image_settings()
    elif settings_str == "cifar_biasfield":
        set_image_settings()
        # Settings.LP_SOLVER = "Gurobi"
    elif settings_str == "mnist_fc":
        set_image_settings()
        Settings.LP_SOLVER = "Gurobi"
        Settings.OVERAPPROX_TYPES = [['deeppoly.area'], ['zono.area'], 
                                ['zono.area', 'zono.ybloat', 'zono.interval', 'deeppoly.area'],
                                ['zono.area', 'zono.ybloat', 'zono.interval', 'deeppoly.area', 'star.lp']] 
        Settings.QUICK_OVERAPPROX_TYPES = [['deeppoly.area'], ['zono.area'],
                                      ['zono.area', 'zono.ybloat', 'zono.interval', 'deeppoly.area']]
    elif "nn4sys" in settings_str:
        set_control_settings()
        Settings.LP_SOLVER = "Gurobi"
        # Settings.OVERAPPROX_TYPES = [['deeppoly.area'], ['zono.area'], 
        #                         ['zono.area', 'zono.ybloat', 'zono.interval', 'deeppoly.area'],
        #                         ['zono.area', 'zono.ybloat', 'zono.interval', 'deeppoly.area', 'star.lp']] 
        # Settings.QUICK_OVERAPPROX_TYPES = [['deeppoly.area'], ['zono.area'],
        #                               ['zono.area', 'zono.ybloat', 'zono.interval', 'deeppoly.area']]
    elif "oval21" in settings_str:
        set_image_settings()
    elif "reach_prob_density" in settings_str:
        set_control_settings()
    elif "rl_benchmarks" in settings_str:
        set_control_settings()
    elif "tllverifybench" in settings_str:
        set_control_settings()
        Settings.LP_SOLVER = "Gurobi"
    elif "vggnet16" in settings_str:
        set_image_settings()
    elif "acasxu" in settings_str:
        set_control_settings()
    elif "collins_rul_cnn" in settings_str:
        set_image_settings()
    elif "cgan" in settings_str:
        set_image_settings()
    elif "metaroom" in settings_str:
        set_image_settings()
        # Settings.LP_SOLVER = "Gurobi"
    elif "soundnessbench" in settings_str:
        set_image_settings()
    elif "relusplitter" in settings_str:
        set_image_settings()
    elif settings_str == "control":
        set_control_settings()
    elif settings_str == "image":
        set_image_settings()
    elif settings_str == "exact":
        set_exact_settings()
    else:
        if num_inputs < 700:
            set_control_settings()
        else:
            set_image_settings()
    # else:
    #     assert settings_str == "exact"
    #     set_exact_settings()

    for init_box, spec in spec_list:
        init_box = np.array(init_box, dtype=input_dtype)

        if timeout is not None:
            if timeout <= 0:
                result_str = 'timeout'
                break

            Settings.TIMEOUT = timeout

        ####################################
        ####################################
        ####################################
        # VNN-COMP 2022 SPECIFIC
        Settings.TIMING_STATS = False
        Settings.PRINT_PROGRESS = False
        ####################################
        ####################################
        ####################################

        res = enumerate_network(init_box, network, spec)
        result_str = res.result_str

        if timeout is not None:
            # reduce timeout by the runtime
            timeout -= res.total_secs

        if result_str != "safe":
            break

    if result_str == "safe":
        result_str = "unsat"
    elif "unsafe" in result_str:
        result_str = "sat"

    if outfile is not None:
        with open(outfile, 'w', encoding="utf-8") as f:
            f.write(result_str)

            if result_str == "sat":
                # print counterexamples
                
                for i, x in enumerate(res.cinput):
                    if i == 0:
                        f.write('\n(')
                    else:
                        f.write('\n')
                    
                    f.write(f"(X_{i} {x})")

                ###########

                for i, y in enumerate(res.coutput):
                    f.write(f"\n(Y_{i} {y})")

                f.write(')')
            
    #print(result_str)

    if result_str == 'error':
        sys.exit(Result.results.index('error'))


if __name__ == '__main__':
    main()