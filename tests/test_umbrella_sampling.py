import pytest
import shutil
from pathlib import Path
from ipy_oxdna.umbrella_sampling import ComUmbrellaSampling, MeltingUmbrellaSampling
from ipy_oxdna.oxdna_simulation import SimulationManager
from unittest.mock import Mock, patch
import numpy as np

class TestSimulation:
    @pytest.fixture
    def us_com(self, tmp_path):
        # Define the path for the file_dir within the temporary directory
        file_dir = tmp_path / "files"

        # Create the file_dir
        file_dir.mkdir()

        # Path to the examples directory in your package
        # Adjust the path as necessary depending on your package structure
        examples_dir = Path(__file__).parent.parent / 'examples' / 'tutorials'/  '8_nt_duplex_melting_cpu' / 'oxdna_files'

        # Files to be copied
        files_to_copy = ['duplex_box_30.top', 'duplex_box_30.dat']

        # Copy each file to the file_dir
        for file_name in files_to_copy:
            shutil.copy(examples_dir / file_name, file_dir / file_name)
            
        sim_dir = file_dir / 'us_com'
        
        us_com = ComUmbrellaSampling(file_dir, sim_dir, clean_build=True)
        
        return us_com
    
    def test_com_umbrella_sampling(self, us_com):
        simulation_manager = SimulationManager()

        com_list = '8,9,10,11,12,13,14,15'
        ref_list = '7,6,5,4,3,2,1,0'
        com_list = ','.join(sorted(com_list.split(','), key=int)[::-1])
        ref_list = ','.join(sorted(ref_list.split(','), key=int))
        xmin = 0
        xmax = 4
        n_windows = 8
        stiff = 3
        temperature = "40C"
        starting_r0 = 0.4213
        print_every = 1e2
        obs_filename = 'com_distance.txt'
        hb_contact_filename = 'hb_contacts.txt'
        pre_eq_steps = 1e3  # This only need to short
        eq_steps = 1e3  # This needs to be long enough to equilibrate the system
        prod_steps = 1e4 # This needs to be long enough to get converged free energy profiles (methods to check are provided)
        particle_indexes = [com_list, ref_list]
        hb_contact_observable = [{'idx':particle_indexes, 'name':f'{hb_contact_filename}', 'print_every':int(print_every)}]
        # oxDNA Simulation parameters
        pre_equlibration_parameters = {
            'backend':'CPU', 'steps':f'{pre_eq_steps}','print_energy_every': f'{pre_eq_steps // 10}',
            'print_conf_interval':f'{pre_eq_steps // 2}', "CUDA_list": "no",'use_edge': 'false',
            'refresh_vel': '1','fix_diffusion': '0', 'T':f'{temperature}'}

        equlibration_parameters = {
            'backend':'CPU','steps':f'{eq_steps}','print_energy_every': f'{eq_steps// 10}',
            'print_conf_interval':f'{eq_steps // 2}', "CUDA_list": "no",'use_edge': 'false',
            'refresh_vel': '1', 'fix_diffusion': '0', 'T':f'{temperature}'}

        production_parameters = {
            'backend':'CPU', 'steps':f'{prod_steps}','print_energy_every': f'{prod_steps}',
            'print_conf_interval':f'{prod_steps}', "CUDA_list": "no", 'use_edge': 'false',
            'refresh_vel': '1','fix_diffusion': '0', 'T':f'{temperature}'}
        
        us_com.build_pre_equlibration_runs(
            simulation_manager, n_windows, com_list, ref_list, stiff, xmin, xmax,
            pre_equlibration_parameters, starting_r0, pre_eq_steps, continue_run=False,
            # If you want to continue a previous simulation set continue_run=int(n_steps)

            print_every=print_every, observable=True, protein=None, sequence_dependant=True,
            force_file=False, name=obs_filename)
            
        assert Path(us_com.pre_equlibration_sim_dir).exists()
        assert len(us_com.pre_equlibration_sims) == n_windows
        
        simulation_manager.worker_manager(cpu_run=True)
        
        for i, sim in enumerate(us_com.pre_equlibration_sims):
            sim.sim_files.parse_current_files()
            assert Path(sim.sim_dir) == Path(us_com.pre_equlibration_sim_dir) / str(i)
            assert Path(sim.sim_files.com_distance).exists()
            assert Path(sim.sim_files.com_distance).stat().st_size > 0
            assert len(Path(sim.sim_files.com_distance).open().readlines()) == 11
        # us_com.analysis.read_all_observables('pre_eq')
        # assert len(us_com.obs_df) == n_windows
        # for df in us_com.obs_df:
        #     assert df.shape == (10,11)
            
        us_com.build_equlibration_runs(
            simulation_manager, n_windows, com_list, ref_list,
            stiff, xmin, xmax, equlibration_parameters, continue_run=False,

            print_every=print_every, observable=True, protein=None, sequence_dependant=True,
            force_file=False, name=obs_filename)
        
        assert Path(us_com.equlibration_sim_dir).exists()
        
        simulation_manager.worker_manager(cpu_run=True)
        
        for i, sim in enumerate(us_com.equlibration_sims):
            sim.sim_files.parse_current_files()
            assert Path(sim.sim_dir) == Path(us_com.equlibration_sim_dir) / str(i)
            assert Path(sim.sim_files.com_distance).exists()
            assert Path(sim.sim_files.com_distance).stat().st_size > 0
            assert len(Path(sim.sim_files.com_distance).open().readlines()) == 11
        
        us_com.build_production_runs(
            simulation_manager, n_windows, com_list, ref_list,
            stiff, xmin, xmax, production_parameters, continue_run=False,

            print_every=print_every, observable=True, protein=None, sequence_dependant=True,
            force_file=False, name=obs_filename)
        
        assert Path(us_com.production_sim_dir).exists()
        
        simulation_manager.worker_manager(cpu_run=True)
        
        for i, sim in enumerate(us_com.production_sims):
            sim.sim_files.parse_current_files()
            assert Path(sim.sim_dir) == Path(us_com.production_sim_dir) / str(i)
            assert Path(sim.sim_files.com_distance).exists()
            assert Path(sim.sim_files.com_distance).stat().st_size > 0
            assert len(Path(sim.sim_files.com_distance).open().readlines()) == 101
        
        
        wham_dir =  Path(__file__).parent.parent / 'wham' / 'wham'
        n_bins = '200'
        tol = '1e-7'
        n_boot = '100'
        us_com.wham_run(wham_dir, xmin, xmax, stiff, n_bins, tol, n_boot)
        
        # us_com.wham.get_n_data_per_com_file()
        # us_com.wham.convergence_analysis(n_chunks, data_added_per_iteration, wham_dir, xmin, xmax, stiff, n_bins, tol, n_boot)
        # temp_range = np.arange(20, 70, 8)
        # n_bins = 50
        # max_hb = 8
        # epsilon = 1e-7
        # convergence_slice = 2
        
        # us_com.discrete_and_continuous_converg_analysis(
        #     convergence_slice, temp_range,
        #     n_bins, xmin, xmax, umbrella_stiff, max_hb,
        #     epsilon=epsilon, reread_files=False)
        
        #assert us_com.pre_equlibration_sim_dir exsits, us_com.equlibration_sim_dir exsits, us_com.production_sim_dir exsits
        
        
        
        
