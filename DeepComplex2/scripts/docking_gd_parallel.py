# -*- coding: utf-8 -*-
"""Docking_GD_Parallel.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1P4GssDPNRBrGwGPIuuTlIdHmC0J2TjxJ
"""

import os
import sys

if len(sys.argv) != 5:
    print('Wrong input parameters\n\n')
    print(len(sys.argv))
    exit()

pdb_path = sys.argv[1]
res_path = sys.argv[2]
OUT = sys.argv[3]
weight_file = sys.argv[4]


from pyrosetta import *
from rosetta import *
from rosetta.protocols.rigid import *
from rosetta.core.scoring import *
from pyrosetta import PyMOLMover
from rosetta.protocols.rigid import *
import pyrosetta.rosetta.protocols.rigid as rigid_moves
from pyrosetta.rosetta.protocols.minimization_packing import MinMover

init()

pmm = PyMOLMover()
pmm.keep_history(True)

working_dir = os.getcwd()


def add_cst(pose, resi, resj, lb, up):
    res_i = pose.pdb_info().pdb2pose('A', res=resi)

    res_j = pose.pdb_info().pdb2pose('B', res=resj)

    if res_i != 0 and res_j != 0:

        atm_i = 'CA' if pose.residue(res_i).name()[0:3] == 'GLY' else 'CB'
        atm_j = 'CA' if pose.residue(res_j).name()[0:3] == 'GLY' else 'CB'

        id_i = AtomID(pose.residue(res_i).atom_index(atm_i), res_i)
        id_j = AtomID(pose.residue(res_j).atom_index(atm_j), res_j)

        ijfunc = rosetta.core.scoring.constraints.BoundFunc(lb, up, 0.1, 'cst1');
        cst_ij = rosetta.core.scoring.constraints.AtomPairConstraint(
            id_i, id_j, ijfunc)

        return cst_ij
    else:
        return False


def add_dist(pose, res_i, res_j):
    if res_i != 0 and res_j != 0:

        atm_i = 'CA' if pose.residue(res_i).name()[0:3] == 'GLY' else 'CB'
        atm_j = 'CA' if pose.residue(res_j).name()[0:3] == 'GLY' else 'CB'

        xyz_i = pose.residue(res_i).xyz(atm_i)
        xyz_j = pose.residue(res_j).xyz(atm_j)

        dist = (xyz_i - xyz_j).norm()

        return dist
    else:
        return False


def detect_diameter(pose):
    import numpy as np

    total = pose.total_residue()
    distances = []

    for i in range(1, total + 1):
        for j in range(1, total + 1):
            dist = add_dist(pose, i, j)
            distances.append(dist)

    return max(distances)


def add_cons_to_pose(pose, res_file):
    protein_diameter = detect_diameter(pose)
    filename = res_file
    with open(filename) as f:
        content = f.readlines()

    lines = [x.rstrip() for x in content]
    cons = []

    for i in range(1, len(lines)):
        data = lines[i].split()
        res_x = int(data[0])
        res_y = int(data[1])
        lb = float(data[2])
        up = float(data[3])
        probability = float(data[4])

        if probability > 0.5:
            if add_cst(pose, res_x, res_y, lb, up) is not False:
                cons.append(add_cst(pose, res_x, res_y, lb, up))

        else:
            if add_cst(pose, res_x, res_y, up, protein_diameter) is not False:
                cons.append(add_cst(pose, res_x, res_y, up, protein_diameter))

    cl = pyrosetta.rosetta.utility.vector1_std_shared_ptr_const_core_scoring_constraints_Constraint_t()
    cl.extend(cons)

    cs = pyrosetta.rosetta.core.scoring.constraints.ConstraintSet()
    cs.add_constraints(cl)

    setup_cons = pyrosetta.rosetta.protocols.constraint_movers.ConstraintSetMover()
    setup_cons.constraint_set(cs)
    setup_cons.apply(pose)


def do_dock(pdb_file, res_file, OUT):
    file_name = os.path.basename(res_file)
    res_name = file_name.split('.')[0]

    target_id = res_name.split('_')[0]

    pose = pyrosetta.pose_from_pdb(pdb_file)
    
    add_cons_to_pose(pose, res_file)

    scorefxn = ScoreFunction()
    scorefxn.add_weights_from_file(weight_file)
    scorefxn.set_weight(atom_pair_constraint, 1)
    sw = SwitchResidueTypeSetMover("centroid")
    switch = SwitchResidueTypeSetMover("fa_standard")

    dock_jump = 1
    partners = 'A_B'
    slide = rosetta.protocols.docking.DockingSlideIntoContact(dock_jump)
    pert_mover = rigid_moves.RigidBodyPerturbMover(dock_jump, 40, 20)

    scorefxn_low = create_score_function("interchain_cen")
    scorefxn_low.set_weight(atom_pair_constraint, 1)
    dock_lowres = rosetta.protocols.docking.DockingLowRes(scorefxn_low, 1)

    sw1 = SwitchResidueTypeSetMover("fa_standard")
    scorefxn_dock = create_score_function("docking", "docking_min")
    scorefxn_dock.set_weight(atom_pair_constraint, 1)

    dock_hires = rosetta.protocols.docking.DockMCMProtocol()
    dock_hires.set_scorefxn(scorefxn_dock)
    dock_hires.set_scorefxn_pack(scorefxn)
    dock_hires.set_partners('A_B')

    movemap = MoveMap()
    movemap.set_jump(dock_jump, True)
    #min_mover = MinMover(movemap, scorefxn, 'lbfgs_armijo_nonmonotone', 0.0001, True)
    min_mover = MinMover(movemap, scorefxn, 'lbfgs', 0.0001, True)
    min_mover.max_iter(1000)

    repeat_mover = RepeatMover(min_mover, 3)

    relax = rosetta.protocols.relax.FastRelax()
    relax.set_scorefxn(scorefxn)
    relax.max_iter(200)
    relax.dualspace(True)
    relax.set_movemap(movemap)

    move_map = MoveMap()
    move_map.set_jump(dock_jump, True)
    move_map.set_chi(True)

    min_mover1 = MinMover(movemap, scorefxn, 'lbfgs', 0.0001, True)
    min_mover1.max_iter(3000)

    jd = PyJobDistributor(target_id, 20, scorefxn)
    temp_pose = Pose()
    temp_pose.assign(pose)
    jd.native_pose = temp_pose

    counter = 0

    while not jd.job_complete:
        test_pose = Pose(pose)
        pmm.apply(test_pose)

        rosetta.protocols.docking.setup_foldtree(test_pose, partners, Vector1([dock_jump]))

        randomize_upstream = RigidBodyRandomizeMover(test_pose, dock_jump, partner_upstream)
        randomize_downstream = RigidBodyRandomizeMover(test_pose, dock_jump, partner_downstream)

        pert_mover.apply(test_pose)
        pmm.apply(test_pose)
        randomize_upstream.apply(test_pose)
        randomize_downstream.apply(test_pose)
        slide.apply(test_pose)

        min_mover.apply(test_pose)
        print(scorefxn.show(test_pose))

        min_mover.apply(test_pose)
        print(scorefxn.show(test_pose))

        min_mover.apply(test_pose)
        print(scorefxn.show(test_pose))

        repeat_mover.apply(test_pose)
        print(scorefxn.show(test_pose))

        switch.apply(test_pose)

        #relax.apply(test_pose)
        min_mover1.apply(test_pose)

        counter = counter + 1
        test_pose.pdb_info().name(target_id + '_' + str(counter))
        jd.output_decoy(test_pose)

        print(scorefxn.show(test_pose))

        score_scorefxn_file = working_dir + '/' + 'score.txt'

        with open(score_scorefxn_file, 'a') as f:
            f.write(test_pose.pdb_info().name())
            f.write(' ')
            f.write(str(scorefxn(test_pose)))
            f.write('\n')

    generated_output = []
    score_results = []

    score_file = working_dir + '/' + target_id + '.fasc'

    with open(score_file) as f:
        for line in f:
            splited_line = line.strip().split(',')
            file_name = splited_line[1].split(':')[1]
            score = splited_line[21].split(':')[1]
            generated_output.append(file_name)
            score_results.append(float(score[:-1]))
            print(file_name, score)

    print(generated_output[score_results.index(min(score_results))])

    pdb_name = generated_output[score_results.index(min(score_results))][2:-1]

    target = pdb_name.split('_')[0]
    best_pdb = working_dir + '/' + pdb_name
    print(best_pdb)
    print(working_dir)
    cmd = "cp " + best_pdb + " " + OUT + "/" + target + "_GD.pdb"
    os.system(cmd)
    print(cmd)
    #cmd = 'rm -rf ' + working_dir + '/' + '*.pdb'
    #os.system(cmd)
    #cmd = 'rm -rf ' + working_dir + '/' + '*.fasc'
    #os.system(cmd)


do_dock(pdb_path, res_path, OUT)
