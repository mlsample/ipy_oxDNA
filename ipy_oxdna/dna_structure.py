from __future__ import annotations

import copy
import json
from collections import namedtuple
from datetime import datetime
from pathlib import Path
from typing import Union, Generator

import numpy as np
from oxDNA_analysis_tools.UTILS.RyeReader import get_traj_info, linear_read
from oxDNA_analysis_tools.UTILS.data_structures import TopInfo, Configuration
from .util import rotation_matrix

# universal indexer for residues
# DOES NOT CORRESPOND TO LINEAR INDEXING IN TOP AND CONF FILES!!!!
# also NO guarentee that all residues will be continuous! or like meaningfully connected at all!
RESIDUECOUNT = 0


def GEN_UIDS(n: int) -> range:
    global RESIDUECOUNT  # hate it
    r = range(RESIDUECOUNT, RESIDUECOUNT + n)
    RESIDUECOUNT += n
    return r


DNABase = namedtuple('DNABase', ['uid', 'base', 'pos', 'a1', 'a3'])


def rc(s):
    mp = {"A": "T", "T": "A", "C": "G", "G": "C"}
    return "".join([mp[c] for c in s][::-1])


def get_rotation_matrix(axis: np.ndarray,
                        angle: Union[int, float, np.float64, np.float32],
                        units: Union[None, str] = None):
    """
    Gets. a rotation matrix around an axis? using rodreguis' formula?
    The main thing this provides is flexability with multiple types
    """
    # if additional angle info provided
    if units:
        if units in ["degrees", "deg", "o"]:
            angle = (np.pi / 180.) * angle  # degrees to radians
            # angle = np.deg2rad (anglest[0])
        elif units == "bp":
            # Notice that the choice of 35.9 DOES NOT correspond to the minimum free energy configuration.
            # This is usually not a problem, since the structure will istantly relax during simulation, but it can be
            # if you need a configuration with an equilibrium value for the linking number.
            # The minimum free energy angle depends on a number of factors (salt, temperature, length, and possibly more),
            # so if you need a configuration with 0 twist make sure to carefully choose a value for this angle
            # and force it in some way (e.g. by changing the angle value below to something else in your local copy).
            # Allow partial bp turns
            angle = angle * (np.pi / 180.) * 35.9
            # Older versions of numpy don't implement deg2rad()
            # angle = int(anglest[0]) * np.deg2rad(35.9)
        # otherwise assume radians
    return rotation_matrix(axis, angle)


class DNAStructureStrand:
    """
    Wrapper class for DNA strand top and conf info
    Note that this object does NOT contain base or strand IDs!
    like oxDNA itself, 3'->5'
    """
    bases: np.ndarray  # char representations of bases
    positions: np.ndarray  # xyz coords of bases
    a1s: np.ndarray  # a1s of bases orientations
    a3s: np.ndarray  # a3s of bases orientations

    def get_a2s(self) -> np.ndarray:
        return np.cross(self.a3s, self.a1s)

    a2s: np.ndarray = property(get_a2s)

    def __init__(self,
                 bases: np.ndarray,
                 positions: np.ndarray,
                 a1s: np.ndarray,
                 a3s: np.ndarray,
                 uids: Union[np.ndarray, None] = None):
        assert bases.size == positions.shape[0] == a1s.shape[0] == a3s.shape[0], "Mismatch in strand raws shapes!"
        assert all([b in ("A", "T", "C", "G") for b in bases]), "Invalid base!"
        assert np.all(~np.isnan(positions))
        if uids is None:
            self.very_global_uids = np.array(GEN_UIDS(len(bases)))
        else:
            self.very_global_uids = uids
        self.bases = bases
        self.positions = positions
        self.a1s = a1s
        self.a3s = a3s

    # def rc(self) -> DNAStructureStrand:
    #     """
    #     TODO: positions, etc.
    #     Returns:
    #         the reverse compliment of this strand
    #     """
    #     #
    #     rev_compl = self[::-1]
    #     # mapping dict
    #     mp = {"A": "T", "T": "A", "C": "G", "G": "C"}
    #
    #     rev_compl.bases = np.array([mp[b] for b in rev_compl.bases])
    #     return rev_compl

    def __getitem__(self, item: Union[int, slice]) -> Union[DNABase, DNAStructureStrand]:
        """
        Returns: a residue object if a position is specified, or a substrand if a slice is specified
        """
        if isinstance(item, int):
            return DNABase(self.very_global_uids[item], self.bases[item], self.positions[item, :], self.a1s[item, :],
                           self.a3s[item, :])
        elif isinstance(item, slice):
            # # sanitize start param
            # if item.start is None:
            #     start = 0
            # else:
            #     start = item.start
            # # sanitize stop param
            # if item.stop is None:
            #     stop = len(self)
            # else:
            #     stop = item.stop
            # assert item.step is None or abs(item.step) == 1, f"Invalid step value {item.step}"
            # # check step
            # if item.step == -1:
            #     # if step is -1, return the strand reversed (implications unclear)
            #     t = start
            #     # gotta lshift to account for [) indexing
            #     start = stop - 1
            #     stop = t - 1
            return DNAStructureStrand(self.bases[item],
                                      self.positions[item, :],
                                      self.a1s[item, :],
                                      self.a3s[item, :],
                                      self.very_global_uids[item])
        else:
            raise Exception(f"Invalid indexer {item}")

    def __len__(self):
        return len(self.bases)

    def assign(self, other: DNAStructureStrand, start: int, stop: int, refr_uids="new"):
        """
        Modifies the strand in-place, substituting the data from the other strand at the start and stop positions
        """
        assert start >= 0, f"Invalid start index {start}"
        assert stop <= len(self), f"Invalid stop index {stop}"
        assert stop > start, f"Invalid indexes {start},{stop}"
        assert start - stop == len(other), f"Length of target region to modify {start - stop} does not match length " \
                                           f" of data source strand object, {len(other)}."
        # assign values
        self.bases[start:stop] = other.bases
        self.positions[start:stop, :] = other.positions
        self.a1s[start:stop, :] = other.a1s
        self.a3s[start:stop, :] = other.a3s
        if refr_uids == "new":
            self.very_global_uids[start:stop] = np.array(GEN_UIDS(stop - start))
        elif refr_uids == "cpy":  # USE WITH CAUTION
            self.very_global_uids[start:stop] = other.very_global_uids

    def append(self, other: Union[DNAStructureStrand, DNABase], cpy_uids=False):
        """
        Depending on the parameter, either concatinates a strand on the 5' of this strand or
        appends a single residue to the end of this strand, in-place
        """
        if isinstance(other, DNABase):
            self.append(DNAStructureStrand(np.array(other.base),
                                           other.pos[np.newaxis, :],
                                           other.a1[np.newaxis, :],
                                           other.a3[np.newaxis, :]))
        elif isinstance(other, DNAStructureStrand):
            self.very_global_uids = np.concatenate([self.very_global_uids, other.very_global_uids if cpy_uids
            else np.array(GEN_UIDS(len(other)))])
            self.bases = np.concatenate([self.bases, other.bases])
            self.positions = np.concatenate([self.positions, other.positions], axis=0)
            self.a1s = np.concatenate([self.a1s, other.a1s], axis=0)
            self.a3s = np.concatenate([self.a3s, other.a3s], axis=0)

    def prepend(self, other: Union[DNAStructureStrand, DNABase], cpy_uids=False):
        """
        Depending on the parameter, either concatinates a strand on the 3' of this strand or
        appends a single residue to the end of this strand, in-place
        """
        if isinstance(other, DNABase):
            self.prepend(DNAStructureStrand(np.array(other.base),
                                            other.pos[np.newaxis, :],
                                            other.a1[np.newaxis, :],
                                            other.a3[np.newaxis, :]))
        elif isinstance(other, DNAStructureStrand):
            self.very_global_uids = np.concatenate([other.very_global_uids if cpy_uids
                                                    else np.array(GEN_UIDS(len(other))), self.very_global_uids])
            self.bases = np.concatenate([other.bases, self.bases])
            self.positions = np.concatenate([other.positions, self.positions], axis=0)
            self.a1s = np.concatenate([other.a1s, self.a1s], axis=0)
            self.a3s = np.concatenate([other.a3s, self.a3s], axis=0)

    def __add__(self, other: Union[DNAStructureStrand, DNABase]):
        """
        Combines a strand with another strand or a DNA base and returns the result, without
        modifying the original strand
        """
        cpy = copy.deepcopy(self)
        cpy.append(other)
        return cpy

    def __iter__(self) -> Generator[DNABase, None, None]:
        # iterate bases in this strand
        for i in range(len(self)):
            yield DNABase(self.very_global_uids[i],
                          self.bases[i],
                          self.positions[i, :],
                          self.a1s[i, :],
                          self.a3s[i, :])

    def transform(self, rot: np.ndarray = np.identity(3), tran: np.ndarray = np.zeros(3)):
        """
        Transforms in place
        """
        assert rot.shape == (3, 3), "Wrong shape for rotation!"
        assert np.linalg.det(rot) - 1 < 1e-9, f"Rotation matrix {rot} has nonzero determinate {np.linalg.det(rot)}"
        assert tran.shape in [(3,), (3, 1)], "Wrong shape for translaton!"
        self.positions = np.einsum('ij, ki -> kj', rot, self.positions) + tran
        self.a1s = np.einsum('ij, ki -> kj', rot, self.a1s)
        self.a3s = np.einsum('ij, ki -> kj', rot, self.a3s)

    def validate(self) -> bool:
        # a1s should all be unit vectors
        if not np.all((np.sum(self.a1s * self.a1s, axis=1) - 1) < 1e-6):
            return False
        # a3s should all be unit vectors
        if not np.all((np.sum(self.a3s * self.a3s, axis=1) - 1) < 1e-6):
            return False
        # all dot products of a1 and a3 should be zero
        if not np.all((np.sum(self.a1s * self.a3s, axis=1)) < 1e-6):
            return False
        # TODO: optional check for distance between adjacent bases?
        return True

    def seq(self, from_5p: bool = False) -> str:
        if from_5p:
            return "".join([*reversed(self.bases)])
        else:
            return "".join(self.bases)


# this shit was originally in a file called helix.py
# i cannot be held resposible for this
BASE_BASE = 0.3897628551303122  # helical distance between two bases? in oxDNA units?
POS_BASE = 0.4  # I think this is the conversion factor from base pairs to oxDNA units????
# gonna say... distance between the helix center of mass of dsDNA to base position?
CM_CENTER_DS = POS_BASE + 0.2
POS_BACK = -0.4
FENE_EPS = 2.0


def strand_from_info(strand_info: list[tuple[chr, np.ndarray, np.ndarray, np.ndarray]]) -> DNAStructureStrand:
    """
    Parameters:
         strand_info (list): a list representing the strand, where each item is a tuple [base, position, a1, a3]

    """
    bases, positions, a1s, a3s = zip(*strand_info)
    return DNAStructureStrand(np.array(bases),
                              np.array(positions),
                              np.array(a1s),
                              np.array(a3s))


def construct_strands(bases: str,
                      start_pos: np.ndarray,
                      helix_direction: np.ndarray,
                      perp: Union[None, bool, np.ndarray] = None,
                      rot: float = 0.,
                      rbases: Union[str, None] = None
                      ) -> tuple[DNAStructureStrand,
                                 DNAStructureStrand]:
    """
    Constructs a pair of antiparallel complimentary DNA strands
    You can discard one of the strands if it isn't needed
    """
    # we need numpy array for these

    # we need to find a vector orthogonal to dir
    # compute magnitude of helix direction vector
    dir_norm = np.sqrt(np.dot(helix_direction, helix_direction))
    # if the vector is zero-length
    if dir_norm < 1e-10:
        # arbitrary default helix direction
        helix_direction = np.array([0, 0, 1])
    else:
        # normalize helix direction vector
        helix_direction /= dir_norm

    if perp is None or perp is False:
        v1 = np.random.random_sample(3)
        v1 -= helix_direction * (np.dot(helix_direction, v1))
        v1 /= np.sqrt(sum(v1 * v1))
    else:
        v1 = perp

    # get bases for sequences
    if rbases is None:
        rbases = rc(bases)
    else:
        assert len(bases) >= len(rbases), "For programmatic reasons, the longer strand should always " \
                                          "be the first arg. I could fix this but don't feel like it"

    # Setup initial parameters
    # and we need to generate a rotational matrix
    # R0 is helix starting rotation
    R0 = get_rotation_matrix(helix_direction, rot)
    # R is the rotation matrix that's applied to each consecutive nucleotide to make the helix a helix
    R = get_rotation_matrix(helix_direction, 1, "bp")
    a1 = v1
    a1 = np.dot(R0, a1)
    # rb = positions along center axis of helix
    rb = np.array(start_pos)
    a3 = helix_direction

    # order here is base, position, a1, a3
    strand_info: list[tuple[chr, np.ndarray, np.ndarray, np.ndarray]] = []
    for i, base in enumerate(bases):
        # compute nucleotide position, a1, a3
        strand_info.append((base, rb - CM_CENTER_DS * a1, a1, a3,))  # , sequence[i]))
        # if this isn't the last base in the sequence
        if i != len(bases) - 1:
            # rotate a1 vector by helix rot
            a1 = np.dot(R, a1)
            # increment position by a3 vector
            rb += a3 * BASE_BASE
    fwd_strand = strand_from_info(strand_info)

    # Fill in complement strand
    strand_info = []  # reuse var
    for ridx, rbase in enumerate(rbases):
        # Note that the complement strand is built in reverse order
        # first compute idx in fwd strand
        i = len(bases) - ridx - 1
        a1 = -fwd_strand[i].a1
        a3 = -fwd_strand[i].a3
        nt2_cm_pos = -(FENE_EPS + 2 * POS_BACK) * a1 + fwd_strand[i].pos
        strand_info.append((rbase, nt2_cm_pos, a1, a3,))
    return fwd_strand, strand_from_info(strand_info)


class DNAStructure:
    """
    This class is intended to import, edit, and export oxDNA confs, since the oat library to
    edit confs is... very very bad
    TODO: incorporate this into OAT
    TODO: alternatively, integerate this into the Scene class

    The class doesn't explicity store base IDs but rather constructs them on-the-fly

    """
    strands: list[DNAStructureStrand]
    time: int
    box: np.ndarray
    energy: np.ndarray
    clustermap: dict[int, int]  # mapping of base uids to cluster indexes
    clusters: list[set[int]]  # list of cluster info, where each cluster is a set of base uids

    def __init__(self,
                 strands: list[DNAStructureStrand],
                 t: int,
                 box: np.ndarray,
                 energy: np.ndarray = np.zeros(3),
                 clusters: list[set[int]] = None):
        self.strands = strands
        self.time = t
        self.box = box
        self.energy = energy
        self.clustermap = dict()
        if clusters is not None:
            self.clusters = clusters
            for i, cluster in enumerate(clusters):
                for uid in cluster:
                    self.clustermap[uid] = i
        else:
            self.clusters = list()

    def get_num_strands(self) -> int:
        return len(self.strands)

    nstrands: int = property(get_num_strands)

    def get_num_bases(self) -> int:
        return sum([len(strand) for strand in self.strands])

    nbases: int = property(get_num_bases)

    def get_base(self, base_idx: int) -> DNABase:
        """
        Returns the residues
        """
        indexer = 0
        for strand_idx, strand in enumerate(self.strands):
            # if the indexer + the length of the strand is greater than the length of the strand
            if indexer + len(strand) > base_idx:
                # return the base in the strand
                return strand[base_idx - indexer]
            # update indexer
            indexer += len(strand)
        raise Exception(f"Base index {base_idx} is greater than total number of bases {indexer}!")

    def base_to_strand(self, base_idx) -> int:
        """
        Returns the index of the strand containing the base with index i
        """
        indexer = 0
        for strand_idx, strand in enumerate(self.strands):
            # if the indexer + the length of the strand is greater than the length of the strand
            if indexer + len(strand) > base_idx:
                # return the strand index
                return strand_idx
            # update indexer
            indexer += len(strand)
        raise Exception(f"Base index {base_idx} is greater than total number of bases {indexer}!")

    def get_strand(self, strand_id) -> DNAStructureStrand:
        return self.strands[strand_id]

    def export_top_conf(self, top_file_path: Path, conf_file_path: Path):
        if not self.has_valid_box():
            self.inbox().export_top_conf(top_file_path, conf_file_path)
        else:
            assert self.validate(), "Invaid structure!"
            with top_file_path.open("w") as topfile, conf_file_path.open("w") as conffile:
                # write conf header
                conffile.write(f't = {int(self.time)}\n')
                conffile.write(f"b = {' '.join(self.box.astype(str))}\n")
                conffile.write(f"E = {' '.join(self.energy.astype(str))}\n")

                # write top file header
                topfile.write(f"{self.nbases} {self.nstrands}\n")
                # loop strands
                base_global_idx = 0
                for sid, strand in enumerate(self.strands):
                    # loop base in strands
                    for base_idx, b in enumerate(strand):
                        # if this strand has a previous base, set n3 to base idx - 1, else set to no previous base (-1)
                        n3 = base_global_idx - 1 if base_idx > 0 else -1
                        # if this strand has a following base, set n5 to base idx + 1, else set to no next base (-1)
                        n5 = base_global_idx + 1 if base_idx < len(strand) - 1 else -1
                        # write residue to top
                        topfile.write(f"{sid + 1} {b.base} {n3} {n5}\n")

                        # write residue to conf
                        # (this class doesn't store particle velocities so skip)
                        pos_str = ' '.join(['{:0.6f}'.format(i) for i in b.pos])
                        a1_str = ' '.join(['{:0.6f}'.format(i) for i in b.a1])
                        a3_str = ' '.join(['{:0.6f}'.format(i) for i in b.a3])
                        conffile.write(f"{pos_str} {a1_str} {a3_str} 0.0 0.0 0.0 0.0 0.0 0.0\n")
                        # conffile.write(
                        #     f"{np.array2string(b.pos)[1:-1]} {np.array2string(b.a1)[1:-1]} {np.array2string(b.a3)[1:-1]} 0.0 0.0 0.0 0.0 0.0 0.0\n")

                        base_global_idx += 1  # iter base global idx

                    # conffile.w
                    # conffile.write('{} {} {} 0.0 0.0 0.0 0.0 0.0 0.0\n'.format(' '.join(p.astype(str)), ' '.join(a1.astype(str)),
                    # ' '.join(a3.astype(str))))

    def poss(self) -> np.ndarray:
        return np.concatenate([s.positions for s in self.strands], axis=0)

    def cms(self) -> np.ndarray:
        """
        Returns the center of mass of this structure
        """

        return np.concatenate([strand.positions for strand in self.strands], axis=0).mean(axis=0)

    def invalidate_box(self):
        self.box = None

    def has_valid_box(self):
        return self.box is not None

    def inbox(self, relpad: Union[float, np.ndarray] = 0.1,
              extrapad: Union[float, np.ndarray] = 0) -> DNAStructure:
        """
        Copies the structure and translates it so that all position values are positive, and resizes the box so
        that all residues are within the box
        """
        if isinstance(extrapad, float):
            extrapad = np.full(extrapad, shape=(3,))
        # get all coordinates
        positions = self.poss()
        # get min and max coords
        mins = np.amin(positions, axis=0)
        maxs = np.amax(positions, axis=0)
        # compute padding
        pad = relpad * (maxs - mins) + extrapad

        # copy self
        cpy = copy.deepcopy(self)
        # translate copy so that mins are at 0
        cpy.transform(tran=-mins + pad)
        # resize box of copy
        cpy.box = maxs - mins + 2 * pad
        new_positions = cpy.poss()
        assert np.all(0 <= new_positions), "Some residues still somehow out of box!"
        assert np.all(new_positions <= cpy.box[np.newaxis, :]), "Some residues still somehow out of box!"
        assert self.validate() == cpy.validate()  # garbage in should imply garbage out and vice versa
        return cpy

    def add_strand(self, strand: DNAStructureStrand):
        self.strands.append(strand)

    def strand_3p(self, strand_idx: int) -> DNABase:
        return self.strands[strand_idx][-1]

    def strand_5p(self, strand_idx: int) -> DNABase:
        return self.strands[strand_idx][0]

    def export_oxview(self, ovfile: Path):
        assert ovfile.parent.exists()
        if not self.has_valid_box():
            self.inbox().export_oxview(ovfile)
        else:
            assert self.validate(), "Invalid structure!"
            oxview_json = {
                "box": self.box.tolist(),
                "date": datetime.now().isoformat(),
                "forces": [],
                "selections": []
            }

            system = {
                "id": 0,
                "strands": []
            }
            bid = 0  # base ID
            for sid, strand in enumerate(self.strands):
                strand_json = {
                    "class": "NucleicAcidStrand",
                    "id": sid,
                    "end3": bid,
                    "end5": bid + len(strand) - 1,
                    "monomers": []
                }
                for base_local_idx, b in enumerate(strand):
                    # if this strand has a previous base, set n3 to base idx - 1, else set to no previous base (-1)
                    n3 = bid - 1 if base_local_idx > 0 else -1
                    # if this strand has a following base, set n5 to base idx + 1, else set to no next base (-1)
                    n5 = bid + 1 if base_local_idx < len(strand) - 1 else -1
                    nucleotide = {
                        "a1": b.a1.tolist(),
                        "a3": b.a3.tolist(),
                        "class": "DNA",
                        "cluster": bid if b.uid not in self.clustermap else self.clustermap[b.uid],
                        "color": 0,  # NOT AN IMPORTANT BIT
                        "id": bid,
                        "n3": n3,
                        "n5": n5,
                        "p": b.pos.tolist(),
                        "type": b.base
                    }
                    strand_json["monomers"].append(nucleotide)
                    #
                    bid += 1
                system["strands"].append(strand_json)
            oxview_json["systems"] = [system]
            with ovfile.open("w") as f:
                json.dump(oxview_json, f)

    def __add__(self, other: Union[DNAStructure, DNAStructureStrand]) -> DNAStructure:
        new_structure = copy.deepcopy(self)
        if isinstance(other, DNAStructureStrand):
            new_structure.strands.append(other)
            return new_structure
        else:
            # ignore time, box, energy, etc. in other
            # TODO: CLUSTERS
            new_structure.strands.extend(other.strands)
            cluster_counter = len(self.clusters)  # starting cluster idxing
            for i, cluster in enumerate(other.clusters):
                for uid in cluster:
                    new_structure.assign_base_to_cluster(uid + self.nbases, i + cluster_counter)
        return new_structure

    def assign_base_to_cluster(self,
                               base_identifier: Union[int, tuple[int, int]],
                               cluster_id: Union[int, None]):
        """
        Assigns a base to a cluster
        """
        assert cluster_id is None or cluster_id <= len(self.clusters)
        if isinstance(base_identifier, tuple):
            uid = self.strands[base_identifier[0]][base_identifier][1].uid
        else:
            uid = base_identifier
        # if cluster id is unspecified or is the length of the cluster list, add a new cluster
        if cluster_id is None or cluster_id == len(self.clusters):
            self.clusters.append(set())
            cluster_id = len(self.clusters) - 1  # handle none values
        if uid in self.clustermap:  # if this base is already assigned a cluster
            self.clusters[self.clustermap[uid]].remove(uid)
        self.clusters[cluster_id].add(uid)
        self.clustermap[uid] = cluster_id

    def transform(self, rot: np.ndarray = np.identity(3), tran: np.ndarray = np.zeros(3)):
        """
        Rotates self in place using the rotation matrix given as an arg
        does NOT update the bounding box!!!
        """
        assert rot.shape == (3, 3), "Wrong shape for rotation!"
        assert abs(np.linalg.det(rot)) - 1 < 1e-9, f"Rotation matrix {rot} has non-one determinate {np.linalg.det(rot)}"
        assert tran.shape in [(3,), (3, 1)], "Wrong shape for translaton!"
        for strand in self.strands:
            strand.transform(rot, tran)
        self.invalidate_box()

    def validate(self) -> bool:
        return all([s.validate() for s in self.strands])


def load_dna_structure(top: Union[str, Path], conf_file: Union[str, Path]) -> DNAStructure:
    """
    Constructs a topology
    """
    if isinstance(top, str):
        top = Path(top)
    if isinstance(conf_file, str):
        conf_file = Path(conf_file)
    # if not top.is_absolute():
    #     top = get_input_dir() / top
    # if not conf_file.is_absolute():
    #     conf_file = get_input_dir() / conf_file

    assert top.is_file(), f"No file exists {str(top)}"
    assert conf_file.is_file(), f"No file exists {str(conf_file)}"

    with top.open("r") as top_file:
        lines = top_file.readlines()
    # get line info
    nbases, nstrands = map(int, lines[0].split())
    # generate placeholder for bases
    # each base represented as a tuple
    strands_list: list[list[tuple[chr, np.ndarray, np.ndarray, np.ndarray]]] = [[] for _ in range(1, nstrands + 1)]
    # generate the return object

    conf: Configuration = next(linear_read(get_traj_info(str(conf_file)),
                                           TopInfo(str(top), nbases)))[0]

    for base_idx, line in enumerate(lines[1:]):
        # find base info
        # strand id, base, next base id, prev base id
        sid, t, p3, p5 = line.split()
        sid, p3, p5 = map(int, [sid, p3, p5])
        # b = Base(t, p3, p5)
        strands_list[sid - 1].append((t,
                                      conf.positions[base_idx, :],
                                      conf.a1s[base_idx, :],
                                      conf.a3s[base_idx, :],))

    strands = [strand_from_info(strand) for strand in strands_list]

    return DNAStructure(strands,
                        conf.time,
                        conf.box,
                        conf.energy)


def load_oxview(oxview: Union[str, Path]):
    if isinstance(oxview, str):
        oxview = Path(oxview)
    # if not oxview.is_absolute():
    #     oxview = get_input_dir() / oxview
    with oxview.open("r") as f:
        ovdata = json.load(f)
        box = ovdata["box"]
        s = DNAStructure([], 0, box)
        # frankly i have no idea how to handle multiple-system files
        if len(ovdata["systems"]) > 1:
            print("Warning: multiple systems will be merged")
        for ox_sys in ovdata["systems"]:
            for strand_data in ox_sys["strands"]:
                if strand_data["class"] == "NucleicAcidStrand":
                    strand = []  # 3' -> 5' list of nucleotides
                    for i, nuc in enumerate(strand_data["monomers"]):
                        a1 = np.array(nuc["a1"])
                        a3 = np.array(nuc["a3"])
                        pos = np.array(nuc["p"])
                        base = nuc["type"]
                        if "n3" in nuc:
                            assert i > 0 and strand_data["monomers"][i - 1]["id"] == nuc["n3"], "Topology problem!"

                        if "n5" in nuc:
                            assert i < len(strand_data["monomers"]) and strand_data["monomers"][i + 1]["id"] == nuc[
                                "n5"], "Topology problem!"
                        strand.append((base, pos, a1, a3))
                    s.add_strand(strand_from_info(strand))
                    # load extra stuff (clusters, etc.)
                    for nuc, nuc_data in zip(s.strands[-1], strand_data["monomers"]):
                        if "cluster" in nuc_data:
                            s.assign_base_to_cluster(nuc.uid, nuc_data["cluster"])

                else:
                    print(f"Unrecognized system data type {strand_data['class']}")
    return s
