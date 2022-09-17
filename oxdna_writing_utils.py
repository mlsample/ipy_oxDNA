from oxdna_json_utils import observables_json
import multiprocessing
import oxpy
import os
import shutil
from json import dumps, loads


def get_dat_top_paths(file_dir):
    os.chdir(file_dir)
    dat_top = os.listdir()
    dat_top_dirs = [file for file in dat_top if (file.endswith(('.dat', '.top'))) and ('trajectory' not in file) and ('last_conf' not in file) and ('energy' not in file)]
    return dat_top_dirs


def get_last_conf_top(file_dir):
    conf_top = os.listdir(file_dir)
    top = [file for file in conf_top if (file.endswith(('.top')))][0]
    try:
        last_conf = [file for file in conf_top if (file.startswith(('last_conf')))][0]
    except IndexError:
        last_conf = [file for file in conf_top if (file.endswith(('.dat'))) and not (file.endswith(('energy.dat'))) and not (file.endswith(('trajectory.dat')))][0]
    return last_conf, top


def write_dat_top(file_dir, sim_dir, dat, top):
    shutil.copy(os.path.join(file_dir,dat), sim_dir)
    shutil.copy(os.path.join(file_dir,top), sim_dir)
    return None


def write_input(sim_dir, json_input_dict, production=False):
    if production is False:
        dat, top = get_last_conf_top(sim_dir)
        json_input_dict["conf_file"] = dat
        json_input_dict["topology"] = top
    #Write input file
    with open(os.path.join(sim_dir, f'input.json'), 'w') as f:
        input_json = dumps(json_input_dict, indent=4)
        f.write(input_json)
    with open(os.path.join(sim_dir, f'input'), 'w') as f:
        with oxpy.Context():
            ox_input = oxpy.InputFile()
            for k, v in json_input_dict.items():
                ox_input[k] = v
            print(ox_input, file=f)

    return None


def get_op_string(begining, end):
    comma = ','
    end = end + 1
    return comma.join(map(lambda x: str(x), range(begining, end)))


def write_op_file(sim_dir, p1, p2):
    p1 = p1.split(',')
    p2 = p2.split(',')
    i = 1
    with open(os.path.join(sim_dir,"op.txt"), 'w') as f:
        f.write("{\norder_parameter = bond\nname = all_native_bonds\n")
    for nuc1 in p1:
        for nuc2 in p2:
            with open(os.path.join(sim_dir,"op.txt"), 'a') as f:
                f.write(f'pair{i} = {nuc1}, {nuc2}\n')
            i += 1
    with open(os.path.join(sim_dir,"op.txt"), 'a') as f:
        f.write("}\n")
    return None


def write_observables_file(sim_dir, p1, p2):
    observe = observables_json(p1, p2, write_path=os.path.join(sim_dir, f'observables.json'))
    return None


def write_run_file(sim_dir, sim_type, run_file):
    with open(run_file, 'r') as f:
        lines = f.readlines()
        with open(os.path.join(sim_dir, 'run.sh'), 'w') as r:
            for line in lines:
                if 'job-name' in line:
                    r.write(f'#SBATCH --job-name="{sim_type}"\n')
                else:
                    r.write(line)
    return None
    


def oxpy_manager(sim_dir):
    with open(os.path.join(sim_dir, 'input.json'), 'r') as f:
        my_input = loads(f.read())
    with oxpy.Context():
        ox_input = oxpy.InputFile()
        for k, v in my_input.items():
            ox_input[k] = v
        manager = oxpy.OxpyManager(ox_input)
        manager.run_complete()
    return None


def spawn(f, args = ()):
    p = multiprocessing.Process(target=f, args=args)
    p.start()
    return p


def run_oxpy_manager(sim_dir):
    p = spawn(oxpy_manager, (sim_dir,))
    print(f'Running {sim_dir}')
    return p