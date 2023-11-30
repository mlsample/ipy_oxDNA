from umbrella_sampling import ComUmbrellaSampling, MeltingUmbrellaSampling
from oxdna_simulation import SimulationManager, Simulation, Observable
import os
import matplotlib.pyplot as plt

def main():

    
    from umbrella_sampling import ComUmbrellaSampling
    from oxdna_simulation import SimulationManager, Simulation
    import os
        
    path = os.path.abspath('../ipy_oxdna_examples')
    file_dir = f'{path}/double_layer'
    system = 'umbrella_0_62_02_gpu_mem_test'
    
    com_list = '3473,3474,3475,3476,3477,3478,3479,3480,3481,3482,3483,3484,3485,3486,3487,3488,3489,3490,3491,3492,3493,3494,3495,3496,3497,3498,3499,3500,3501,3502,3503,3504,3505,3506,3507,3508,3509,3510,3511,3512,3513,3514,3515,3516,3517,3518,3519,3520,3521,3522,3523,3524,3525,3526,3527,3528,3529,3530,3531,3532,3533,3534,3535,3536,3537,3538,3539,3540,3541,3542,3543,3544,3545,3546,3547,3548,3549,3550,3551,3552,3553,3554,3555,3556,3557,3558,3559,3560,3561,3562,3563,3564,3565,3566,3567,3568,3569,3570,3571,3572,3573,3574,3575,3576,3577,3578,3579,3580,3581,3582,3583,3584,3585,3586,3587,3588,3589,3590,3591,3592,3593,3594,3595,3596,3597,3598,3599,3600,3601,3602,3603,3604'
    ref_list = '6913,6914,6915,6916,6917,6918,6919,6920,6921,6922,6923,6924,6925,6926,6927,6928,6929,6930,6931,6932,6933,6934,6935,6936,6937,6938,6939,6940,6941,6942,6943,6944,6945,6946,6947,6948,6949,6950,6951,6952,6953,6954,6955,6956,6957,6958,6959,6960,6961,6962,6963,6964,6965,6966,6967,6968,6969,6970,6971,6972,6973,6974,6975,6976,6977,6978,6979,6980,6981,6982,6983,6984,6985,6986,6987,6988,6989,6990,6991,6992,6993,6994,6995,6996,6997,6998,6999,7000,7001,7002,7003,7004,7005,7006,7007,7008,7009,7010,7011,7012,7013,7014,7015,7016,7017,7018,7019,7020,7021,7022,7023,7024,7025,7026,7027,7028,7029,7030,7031,7032,7033,7034,7035,7036,7037,7038,7039,7040,7041,7042,7043,7044'
    
    xmin = 0
    xmax = 72.787
    n_windows = 100
    
    stiff = 0.2
    
    
    equlibration_parameters = {'steps':'1e7', 'T':'20C', 'print_energy_every': '1e6', 'print_conf_interval':'1e7', 'max_density_multiplier':'1.5'}
    production_parameters = {'steps':'2e7', 'T':'20C', 'print_energy_every': '2e6', 'print_conf_interval':'2e7', 'max_density_multiplier':'1.5'}
    
    us = ComUmbrellaSampling(file_dir, system)
    simulation_manager = SimulationManager()
    
    us.build_equlibration_runs(simulation_manager, n_windows, com_list, ref_list, stiff, xmin, xmax, equlibration_parameters, observable=False)
    simulation_manager.worker_manager()
    
    us.build_production_runs(simulation_manager, n_windows, com_list, ref_list, stiff, xmin, xmax, production_parameters, observable=True, print_every=1e4, name='com_distance.txt')   
    simulation_manager.worker_manager()

if __name__ == '__main__':
    main()