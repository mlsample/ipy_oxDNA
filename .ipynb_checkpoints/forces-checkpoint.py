import os

class Force:
    @staticmethod
    def com_force(p1=None, p2=None, k=None, r0=None, pbc=None, rate=None):
        external_force = 'com'
        com_force_parameters = {"external_force":external_force,
                                "p1": p1,
                                "p2": p2,
                                "k": k,
                                "r0": r0,
                                "pbc": pbc,
                                "rate": rate
                                }
        return com_force_parameters


    @staticmethod
    def mutual_trap(particle=None, pos0=None, k=None, pbc=None, rate=None, dir=None):
        external_force = 'trap'
        mutual_trap_parameters = {"external_force":external_force,
                                  "particle": particle,
                                  "pos0": pos0,
                                  "k": k,
                                  "pbc": pbc,
                                  "rate": rate,
                                  "dir": dir
                                  }
        return mutual_trap_parameters
