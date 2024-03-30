class DefaultInput:
    _input: dict[str, str]

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

    def get_dict(self) -> dict[str, str]:
        """
        Returns: the values
        """
        return {
            key: str(self._input[key]) for key in self._input
        }

    def __getitem__(self, item: str) -> str:
        return str(self._input[item])


# todo: better
SEQ_DEP_PARAMS: dict[str, float] = {
    "STCK_G_C": 1.69339,
    "STCK_C_G": 1.74669,
    "STCK_G_G": 1.61295,
    "STCK_C_C": 1.61295,
    "STCK_G_A": 1.59887,
    "STCK_T_C": 1.59887,
    "STCK_A_G": 1.61898,
    "STCK_C_T": 1.61898,
    "STCK_T_G": 1.66322,
    "STCK_C_A": 1.66322,
    "STCK_G_T": 1.68032,
    "STCK_A_C": 1.68032,
    "STCK_A_T": 1.56166,
    "STCK_T_A": 1.64311,
    "STCK_A_A": 1.84642,
    "STCK_T_T": 1.58952,
    "HYDR_A_T": 0.88537,
    "HYDR_T_A": 0.88537,
    "HYDR_C_G": 1.23238,
    "HYDR_G_C": 1.23238
}

NA_PARAMETERS = {
    "HYDR_A_U": 1.21,
    "HYDR_A_T": 1.37,
    "HYDR_rC_dG": 1.61,
    "HYDR_rG_dC": 1.77
}

RNA_PARAMETERS = {
    "HYDR_A_T": 0.820419,
    "HYDR_C_G": 1.06444,
    "HYDR_G_T": 0.510558,
    "STCK_G_C": 1.27562,
    "STCK_C_G": 1.60302,
    "STCK_G_G": 1.49422,
    "STCK_C_C": 1.47301,
    "STCK_G_A": 1.62114,
    "STCK_T_C": 1.16724,
    "STCK_A_G": 1.39374,
    "STCK_C_T": 1.47145,
    "STCK_T_G": 1.28576,
    "STCK_C_A": 1.58294,
    "STCK_G_T": 1.57119,
    "STCK_A_C": 1.21041,
    "STCK_A_T": 1.38529,
    "STCK_T_A": 1.24573,
    "STCK_A_A": 1.31585,
    "STCK_T_T": 1.17518,
    "CROSS_A_A": 59.9626,
    "CROSS_A_T": 59.9626,
    "CROSS_T_A": 59.9626,
    "CROSS_A_C": 59.9626,
    "CROSS_C_A": 59.9626,
    "CROSS_A_G": 59.9626,
    "CROSS_G_A": 59.9626,
    "CROSS_G_G": 59.9626,
    "CROSS_G_C": 59.9626,
    "CROSS_C_G": 59.9626,
    "CROSS_G_T": 59.9626,
    "CROSS_T_G": 59.9626,
    "CROSS_C_C": 59.9626,
    "CROSS_C_T": 59.9626,
    "CROSS_T_C": 59.9626,
    "CROSS_T_T": 59.9626,
    "ST_T_DEP": 1.97561
}
