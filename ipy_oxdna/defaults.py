class DefaultInput:
    def __init__(self):
        self.cuda_MD()
        
    def swap_default_input(self, default_type: str):
        if default_type == "cuda_MD":
            self.cuda_MD()
        elif default_type == "cpu_MD":
            self.cpu_MD()
        elif default_type == "cpu_MC_relax":
            self.cpu_MC_relax()
        else:
            raise ValueError("Invalid default_type")
    
    def cuda_MD(self):
        self._input = {
            "interaction_type": "DNA2",
            "salt_concentration": "1.0",
            "sim_type": "MD",
            "backend": "CUDA",
            "backend_precision": "mixed",
            "use_edge": "1",
            "edge_n_forces": "1",
            "CUDA_list": "verlet",
            "CUDA_sort_every": "0",
            "max_density_multiplier": "3",
            "steps": "1e9",
            "ensemble": "nvt",
            "thermostat": "john",
            "T": "20C",
            "dt": "0.003",
            "verlet_skin": "0.5",
            "diff_coeff": "2.5",
            "newtonian_steps": "103",
            "topology": None,
            "conf_file": None,
            "lastconf_file": "last_conf.dat",
            "trajectory_file": "trajectory.dat",
            "refresh_vel": "1",
            "no_stdout_energy": "0",
            "restart_step_counter": "1",
            "energy_file": "energy.dat",
            "print_conf_interval": "5e5",
            "print_energy_every": "5e5",
            "time_scale": "linear",
            "max_io": "5",
            "external_forces": "0",
            "external_forces_file": "forces.json",
            "external_forces_as_JSON": "true"
        }
    
    def cpu_MD(self):
        self._input = {
            "interaction_type": "DNA2",
            "salt_concentration": "1.0",
            "sim_type": "MD",
            "backend": "CPU",
            "backend_precision": "double",
            "steps": "1e8",
            "ensemble": "nvt",
            "thermostat": "john",
            "T": "20C",
            "dt": "0.003",
            "verlet_skin": "0.5",
            "diff_coeff": "2.5",
            "newtonian_steps": "103",
            "topology": None,
            "conf_file": None,
            "lastconf_file": "last_conf.dat",
            "trajectory_file": "trajectory.dat",
            "refresh_vel": "1",
            "no_stdout_energy": "0",
            "restart_step_counter": "1",
            "energy_file": "energy.dat",
            "print_conf_interval": "5e5",
            "print_energy_every": "5e5",
            "time_scale": "linear",
            "max_io": "5",
            "external_forces": "0",
            "external_forces_file": "forces.json",
            "external_forces_as_JSON": "true"
        }
    
    def cpu_MC_relax(self):
        self._input = {
            "sim_type": "MC",
            "backend": "CPU",
            "backend_precision": "double",
            "verlet_skin": "0.5",
            "interaction_type": "DNA2",
            "steps": "5e3",
            "dt": "0.05",
            "T": "30C",
            "salt_concentration": "1.0",
            "ensemble": "nvt",
            "delta_translation": "0.22",
            "delta_rotation": "0.22",
            "diff_coeff": "2.5",
            "max_backbone_force": "5",
            "max_backhone_force_far": "10",
            "topology": None,
            "conf_file": None,
            "lastconf_file": "last_conf.dat",
            "trajectory_file": "trajectory.dat",
            "energy_file": "energy.dat",
            "print_conf_interval": "${$(steps) / 10}",
            "print_energy_every": "${$(steps) / 50}",
            "time_scale": "linear",
            "refresh_vel": "1",
            "restart_step_counter": "1",
            "no_stdout_energy": "0",
            "max_io": "5",
            "external_forces": "0",
            "external_forces_file": "forces.json",
            "external_forces_as_JSON": "true"
        }