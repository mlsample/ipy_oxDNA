oxdna\_simulation module
========================

.. autoclass:: oxdna_simulation.Simulation
   :members: build, input_file, sequence_dependant, add_force_file, add_protein_par, add_force, add_observable

.. autoclass:: oxdna_simulation.SimulationManager
   :members: start_nvidia_cuda_mps_control, run, worker_manager, queue_sim
   
.. autoclass:: oxdna_simulation.GenerateReplicas
   :members: multisystem_replica, concat_all_system_traj

.. autoclass:: oxdna_simulation.OxdnaAnalysisTools
   :members: 
   
.. autoclass:: oxdna_simulation.Analysis
   :members: get_init_conf, view_init, get_last_conf, view_last, get_conf, view_conf, get_conf_count, current_step, plot_energy, plot_observable, hist_observable

.. autoclass:: oxdna_simulation.Force
   :members:

.. autoclass:: oxdna_simulation.Observable
   :members: