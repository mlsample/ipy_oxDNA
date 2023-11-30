from umbrella_sampling import ComUmbrellaSampling, MeltingUmbrellaSampling
from oxdna_simulation import SimulationManager, Simulation, Observable
import os
import matplotlib.pyplot as plt
import timeit


def main(): 
    path = os.path.abspath('../ipy_oxdna_examples/nanotube/104')
    system_name = ['514']
    file_dirs = [f'{path}/{sys}' for sys in system_name]
    systems = [f'{sys}_umbrella' for sys in system_name]
    
    com_list = '3566,3774,3567,3568,3569,3570,3571,3572,3573,3574,3575,3576,3577,3578,3579,3580,3581,3582,3583,3584,3585,3586,3587,3588,3589,3590,3591,3592,3593,3594,3595,3596,3597,3598,3599,3600,3601,3602,3603,3604,3605,3606,3607,3608,3609,3610,3611,3612,3613,3614,3615,3616,3617,3618,3619,3620,3621,3622,3623,3624,3625,3626,3627,3628,3629,3630,3631,3632,3633,3634,3635,3636,3637,3638,3639,3640,3641,3642,3643,3644,3645,3646,3647,3648,3649,3650,3651,3652,3653,3654,3655,3656,3657,3658,3659,3660,3661,3662,3663,3664,3665,3666,3667,3668,3669,3670,3671,3672,3673,3674,3675,3676,3677,3678,3679,3680,3681,3682,3683,3684,3685,3686,3687,3688,3689,3690,3691,3692,3693,3694,3695,3696,3697,3698,3699,3700,3701,3702,3703,3704,3705,3706,3707,3708,3709,3710,3711,3712,3713,3714,3715,3716,3717,3718,3719,3720,3721,3722,3723,3724,3725,3726,3727,3728,3729,3730,3731,3732,3733,3734,3735,3736,3737,3738,3739,3740,3741,3742,3743,3744,3745,3746,3747,3748,3749,3750,3751,3752,3753,3754,3755,3756,3757,3758,3759,3760,3761,3762,3763,3764,3765,3766,3767,3768,3769,3770,3771,3772,3773'
                            
    ref_list = '7163,6966,6967,6968,6969,6970,6971,6972,6973,6974,6975,6976,6977,6978,6979,6980,6981,6982,6983,6984,6985,6986,6987,6988,6989,6990,6991,6992,6993,6994,6995,6996,6997,6998,6999,7000,7001,7002,7003,7004,7005,7006,7007,7008,7009,7010,7011,7012,7013,7014,7015,7016,7017,7018,7019,7020,7021,7022,7023,7024,7025,7026,7027,7028,7029,7030,7031,7032,7033,7034,7035,7036,7037,7038,7039,7040,7041,7042,7043,7044,7045,7046,7047,7048,7049,7050,7051,7052,7053,7054,7055,7056,7057,7058,7059,7060,7061,7062,7063,7064,7065,7066,7067,7068,7069,7070,7071,7072,7073,7074,7075,7076,7077,7078,7079,7080,7081,7082,7083,7084,7085,7086,7087,7088,7089,7090,7091,7092,7093,7094,7095,7096,7097,7098,7099,7100,7101,7102,7103,7104,7105,7106,7107,7108,7109,7110,7111,7112,7113,7114,7115,7116,7117,7118,7119,7120,7121,7122,7123,7124,7125,7126,7127,7128,7129,7130,7131,7132,7133,7134,7135,7136,7137,7138,7139,7140,7141,7142,7143,7144,7145,7146,7147,7148,7149,7150,7151,7152,7153,7154,7155,7156,7157,7158,7159,7160,7161,7162,7164,7165,7166,7167,7168,7169,7170,7171,7172,7173,7174'
                
                
    
    stiff = 0.2
    xmin = 0
    xmax = 72.787
    n_windows = 47
    
    equlibration_parameters = {'steps':'1e7', 'print_energy_every': '1e7', 'print_conf_interval':'1e7'}
    production_parameters = {'steps':'2e7', 'print_energy_every': '2e7', 'print_conf_interval':'2e7'}
    
    us_list = [ComUmbrellaSampling(file_dir, sys) for file_dir, sys in zip(file_dirs,systems)]
    
    simulation_manager = SimulationManager()
        
    for us in us_list:
        us.build_equlibration_runs(simulation_manager, n_windows, com_list, ref_list, stiff, xmin, xmax, equlibration_parameters, observable=False)
    simulation_manager.worker_manager()
        
    for us in us_list:
        us.build_production_runs(simulation_manager, n_windows, com_list, ref_list, stiff, xmin, xmax, production_parameters, observable=True, print_every=1e3, name='com_distance.txt')
    simulation_manager.worker_manager()
    
    wham_dir = os.path.abspath('/scratch/mlsample/ipy_oxDNA/wham/wham')
    n_bins = '200'
    tol = '1e-7'
    n_boot = '100'
    for us in us_list:
        us.wham_run(wham_dir, xmin, xmax, stiff, n_bins, tol, n_boot)

if __name__ == '__main__':
    tic = timeit.default_timer()
    main()
    toc = timeit.default_timer()
    print(f'Umbrella run time: {toc - tic}')
