#!/home/mariaharris/.conda/envs/my-rdkit-env/bin/python

"""
Runs RMSD-PP calculations to get a TS guess structure and barrier estimate for the reaction defined
by the input reactant and product structures. 

The RMSD-PP procedure is part of the xTB program and described in

    Stefan Grimme
    "Exploration of Chemical Compound, Conformer, and
    Reaction Space with Meta-Dynamics Simulations Based on Tight-Binding
    Quantum Chemical Calculations"
    J. Chem. Theory Comput. 2019, Vol. 15, 2847-2862
    DOI: 10.1021/acs.jctc.9b00143
"""


import os
import sys
import itertools
import shutil
import time
import tarfile
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import multiprocessing as mp
import seaborn as sns
import xyz2mol_local
from rdkit import Chem
from rdkit import RDLogger
from scipy.signal import argrelextrema
from contextlib import redirect_stdout
from contextlib import redirect_stderr


def submit_xtb_path_calculation(reactant_xyz, product_xyz, k_push, k_pull, alp):
    """
    This function submits an xtb path job from reactant_xyz to product_xyz
    using the given k_pull, k_push and alpha values. The "path.inp" file should
    be in the current directory.
    It relies on the use of 'submit_xtb_path', which is a bash submit script
    calling the xTB program and submits it to slurm which returns a jobid which
    can be used to monitor the job.
    """
    jobid = os.popen('submit_xtb_path ' + reactant_xyz + ' ' + product_xyz + \
    ' ' + str(k_push) + ' ' + str(k_pull) + ' ' + str(alp)).read()
    jobid = int(jobid.split()[-1])
    return {jobid}


def check_if_reaction_complete(out_file):
    """
    From xtb path calculation where three runs have been done. Based on the
    RMSE between end structure and target structure it is determined whether
    the reaction has been completed.
    Returns list of "True" and "False" for the three reactions depending on
    whether the reacion was finished or not, respectively.
    """
    barriers = []
    rmse_prod_to_endpath = []
    reactions_completed = []
    with open(out_file, 'r') as _file:
        line = _file.readline()
        while not barriers:
            if "energies in kcal/mol, RMSD in Bohr" in line:
                for _ in range(3):
                    line = _file.readline()
                    data = line.split()
                    if data[0] == "WARNING:":
                        line = _file.readline()
                        data = line.split()
                    barriers.append(np.float(data[3]))
                    rmsd = np.float(data[14])
                    rmse_prod_to_endpath.append(rmsd)
                    if rmsd < 0.3:
                        reactions_completed.append(True)
                    else:
                        reactions_completed.append(False)
            line = _file.readline()

    return reactions_completed

def check_activity_of_bonds(xyz_file, bond_pairs):
    """
    This function checks whether the bonds being formed or broken fulfills the
    "activity" criteria that 1.2 =< r_ij/(r_cov,i+r_cov,j) =< 1.7
    where r_ij is the bond distance of bond pair i,j in the structure in
    xyz_file. r_cov,i is the covalent distance of atom i. This criteria is
    taken from atom_mapper paper.
    """
    active_bonds = []
    atom_numbers_list, coordinates_list, _ = get_coordinates([xyz_file])
    ptable = Chem.GetPeriodicTable()
    coordinates = coordinates_list[0]
    atom_numbers = atom_numbers_list[0]
    for bond_pair in bond_pairs:
        atom_i = atom_numbers[bond_pair[0]]
        atom_j = atom_numbers[bond_pair[1]]
        print(atom_i, atom_j)
        r_distance =  \
        np.linalg.norm(coordinates[bond_pair[0], :]-coordinates[bond_pair[1], :])
        r_cov_i = ptable.GetRcovalent(atom_i)
        r_cov_j = ptable.GetRcovalent(atom_j)
        bond_activity = r_distance/(r_cov_i+r_cov_j)
        print('Bond activity for ', bond_pair, ' = '+str(bond_activity))
        if 1.2 <= bond_activity <= 1.7:
            print("bond active")
            active_bonds.append(True)
        else:
            print("bond not active")
            active_bonds.append(False)
    return active_bonds

def get_relaxed_xtb_structure(xyz_path, new_file_name):
    """
    choose the relaxed structure from the last point on the xtb path
    """
    with open(xyz_path, 'r') as _file:
        line = _file.readline()
        n_lines = int(line.split()[0])+2
    count = 0
    input_file = open(xyz_path, 'r')
    dest = None
    for line in input_file:
        if count % n_lines == 0:
            if dest:
                dest.close()
            dest = open(new_file_name, "w")
        count += 1
        dest.write(line)

def extract_xtb_structures(path_file):
    """
    Extract the structures of the path calculated by xtb and save them in a
    directory to be submitted to single point calculations (xtb_sp)
    """

    dat_file = path_file[:-6]+'.dat'
    n_run = int(path_file[-5])
    os.mkdir("xtb_sp")
    xtb_coordinates = []
    with open(path_file, 'r') as _file:
        line = _file.readline()
        n_lines = int(line.split()[0])+2

    with open(dat_file, 'r') as _file:
        lines = _file.read()

    paths = lines.split('\n \n')
    path = paths[n_run-1]
    path = path.split('\n')
    for line in path:
        data = line.split()
        coordinate = np.float(data[0])
        xtb_coordinates.append(coordinate)

    count = 0
    indx = 0
    dest = None
    with open(path_file, 'r') as _file:
        for line in _file:
            if count % n_lines == 0:
                if dest:
                    dest.close()
                dest = open('xtb_sp/'+str(indx)+'.xyz', 'w')
                indx += 1
            dest.write(line)
            count += 1
        dest.close()

    return xtb_coordinates



def submit_xtb_sp(directory):
    """
    submits all .xyz files in directory and calculates single point energies
    using xtb.
    It relies on the existence of the bash submit script 'submit_batches_xtb'
    which submits a batch of single-point energy calculations to the xTB
    program to slurm, returning a jobid which can be used to monitor the job.
    """
    os.chdir(directory)
    jobid = os.popen('submit_batches_xtb').read()
    jobid = set([int(jobid.split()[-1])])
    os.chdir("../")
    return jobid




def get_coordinates(structure_files_list):
    """
    Extrapolate around maximum structure on the xtb surface to make DFT single
    point calculations in order to choose the best starting point for TS
    optimization. Should return this starting point structure
    """
    n_structures = len(structure_files_list)
    atom_numbers_list = []
    coordinates_list = []
    for i in range(n_structures):
        atom_numbers = []
        with open(structure_files_list[i], 'r') as struc_file:
            line = struc_file.readline()
            n_atoms = int(line.split()[0])
            struc_file.readline()
            coordinates = np.zeros((n_atoms, 3))
            for j in range(n_atoms):
                line = struc_file.readline().split()
                atom_number = line[0]
                atom_numbers.append(atom_number)
                coordinates[j, :] = np.array([np.float(num) for num in
                                              line[1:]])
        atom_numbers_list.append(atom_numbers)
        coordinates_list.append(coordinates)
    return atom_numbers_list, coordinates_list, n_atoms



def make_sp_interpolation(atom_numbers_list, coordinates_list, n_atoms, n_points):
    """
    From the given structures in coordinates_list xyz files are created by
    extrapolating between those structures with n_points between each structure
    creates a directory "path" with those .xyz files
    """
    n_structures = len(coordinates_list)
    os.mkdir('path')
    with open('path/path_file.txt', 'w') as path_file:
        for i in range(n_structures-1):
            difference_mat = coordinates_list[i+1]-coordinates_list[i]
            for j in range(n_points+1):
                path_xyz = coordinates_list[i]+j/n_points*difference_mat
                path_xyz = np.matrix(path_xyz)
                file_path = 'path/path_point_' + str(i*n_points+j)+'.xyz'
                with open(file_path, 'w+') as _file:
                    _file.write(str(n_atoms)+'\n\n')
                    path_file.write(str(n_atoms)+'\n\n')
                    for atom_number, line in zip(atom_numbers_list[i], path_xyz):
                        _file.write(atom_number+' ')
                        path_file.write(atom_number+' ')
                        np.savetxt(_file, line, fmt='%.6f')
                        np.savetxt(path_file, line, fmt='%.6f')



def make_gaussian_sp(xyz_file, method):
    """
    This function prepares a Gaussiam input file for a single point
    calculation with method B3LYP/6-31G(d,p)
    """
    com_file = xyz_file[:-4]+'_sp.com'
    with open(com_file, 'w') as _file:
        _file.write('%mem=16GB'+'\n')
        _file.write('%nprocshared=4'+'\n')
        _file.write('#'+str(method)+' scf=(maxcycles=1024)'+'\n\n')
        _file.write('something title \n\n')
        _file.write('0 1 \n')
        with open(xyz_file, 'r') as file_in:
            lines = file_in.readlines()
            lines = lines[2:]
            _file.writelines(lines)
        _file.write('\n')

    return com_file

def submit_gaussian_sp(directory, method):
    """
    make sp com files on all files on directory
    It relies on the existence of the bash submit script
    'submit_batches_gaussian', which submits a batch of single-point energy
    calculations to the Gaussian16 program to slurm, returning a jobid which can be used
    to monitor the job.

    """
    os.chdir(directory)
    files = os.listdir(os.curdir)
    for _file in files:
        if _file.endswith(".xyz"):
            com_file = make_gaussian_sp(_file, method)
    jobid = os.popen('submit_batches_gaussian').read()
    jobid = set([int(jobid.split()[-1])])
    os.chdir("../")
    return jobid

def wait_for_jobs_to_finish(job_ids):
    """
    This script checks with slurm if a specific set of jobids is finished with a
    frequency of 1 minute.
    Stops when the jobs are done.
    """
    while True:
        job_info1 = os.popen("squeue -p mko").readlines()[1:]
        job_info2 = os.popen("squeue -u mariaharris").readlines()[1:]
        current_jobs1 = {int(job.split()[0]) for job in job_info1}
        current_jobs2 = {int(job.split()[0]) for job in job_info2}
        current_jobs = current_jobs1|current_jobs2
        if current_jobs.isdisjoint(job_ids):
            break
        else:
            time.sleep(60)

def plot_path(x_list, y_list, _color, _label, x_label):
    """
    plots the values in y_list versus those in x_list
    """
    sns.set()
    plt.plot(x_list, y_list, marker='o', color=_color, label=_label)
    plt.xlabel(x_label)
    plt.ylabel('Energy [kcal/mol]')
    plt.legend()


def find_xtb_max_from_sp_interpolation(directory, extract_max_structures):
    """
    when sp calculations ar efinished: find the structure with maximum xtb
    energy
    """
    energies = []
    path_points = []

    os.chdir(directory)
    tfs = [f for f in os.listdir(os.curdir) if \
            f.endswith("xtb.tar.gz")]

    for file_name in tfs:
        tf = tarfile.open(file_name, mode='r')
        tf.extractall()
        tf.close()

    files = [f for f in os.listdir(os.curdir) if \
             f.endswith("xtbout")]
    files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    for file_name in files:
        path_point = int(''.join(filter(str.isdigit, file_name)))
        path_points.append(path_point)
        with open(file_name, 'r') as _file:
            line = _file.readline()
            while line:
                if 'TOTAL ENERGY' in line:
                    energy_au = np.float(line.split()[3])
                line = _file.readline()
        energies.append(energy_au)
        os.remove(file_name)
    energies_kcal = np.array(energies)*627.509
    energies_kcal = energies_kcal-energies_kcal[0]
    max_index = energies.index(max(energies))
    if extract_max_structures:
        max_point = path_points[max_index]
        shutil.copy(str(max_point-1)+'.xyz', '../max_structure-1.xyz')
        shutil.copy(str(max_point)+'.xyz', '../max_structure.xyz')
        shutil.copy(str(max_point+1)+'.xyz', '../max_structure+1.xyz')
    os.chdir("../")
    return max(energies), energies_kcal, max_index

def find_dft_max_from_sp_interpolation(directory):
    """
    When the sp calculations are finished: find the structure with maximum
    energy to prepare for a TS calculation
    """
    energies = []
    path_points = []

    os.chdir(directory)
    tfs = [f for f in os.listdir(os.curdir) if \
            f.endswith("gaus.tar.gz")]

    for file_name in tfs:
        tf = tarfile.open(file_name, mode='r')
        tf.extractall()
        tf.close()

    files = [f for f in os.listdir(os.curdir) if\
            f.endswith("sp.out")]
    files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    for file_name in files:
        path_point = int(''.join(filter(str.isdigit, file_name)))
        path_points.append(path_point)
        with open(file_name, 'r') as _file:
            line = _file.readline()
            while line:
                if 'SCF Done' in line:
                    energy_au = np.float(line.split()[4])
                line = _file.readline()
        energies.append(energy_au)
        os.remove(file_name)
    print("Max DFT energy from sp interpolation = "+str(max(energies)))
    max_index = energies.index(max(energies))
    max_point = path_points[max_index]
    ts_test_file = str(directory)+"/path_point_"+str(max_point)+".xyz"
    os.chdir("../")
    return ts_test_file, energies



def get_adjacency_matrix(structure_file):
    """
    Using functions in the xyz2mol module the adjacency matrix of the structure
    in structure_file is calculated and returned
    """

    atoms, _, xyz_coordinates = \
    xyz2mol_local.read_xyz_file(structure_file)

    adjacency_matrix, _ = xyz2mol_local.xyz2AC(atoms, xyz_coordinates, 0,
                                               use_huckel=True)

    return adjacency_matrix



def find_bonds_getting_formed_or_broken(reaction_structure, product_structure):
    """
    Based on the reaction and product structure, the bonds that are
    fomed/broken are singled out for contraintment
    the difference in the afjacency matric tells whether bond has been formed
    (+1) or bond is broken (-1)
    """

    bond_pairs_changed = []
    reaction_ac = get_adjacency_matrix(reaction_structure)
    product_ac = get_adjacency_matrix(product_structure)
    n_atoms = len(reaction_ac)

    difference_mat = product_ac - reaction_ac
    for combination in itertools.combinations(range(n_atoms), 2):
        combination = list(combination)
        bond_change = difference_mat[combination[0], combination[1]]
        if bond_change != 0:
            bond_pairs_changed.append(combination)
    print(bond_pairs_changed)
    return bond_pairs_changed



def make_gaussian_constrained_opt(xyz_file, bond_pairs_changed,
                                  method):
    """
    This function prepares a Gaussian input file for a constrained
    optimization freezing the bonds broken or created during the reaction.
    The .com file is submitted to slurm using 'submit_gaus16' which returns a
    jobid used to monitor the job
    """
    n_frozen_bonds = len(bond_pairs_changed)
    com_file = xyz_file[:-4]+'_opt.com'
    with open(com_file, 'w') as _file:
        _file.write('%mem=16GB'+'\n')
        _file.write('%nprocshared=4'+'\n')
        _file.write('#opt '+str(method)+' geom(modredundant, gic) ')
        _file.write('scf=(maxcycles=1024)'+'\n\n')
        _file.write('something title'+'\n\n')
        _file.write('0 1'+'\n') #hardcoded to neutral singlet, change if needed
        with open(xyz_file, 'r') as file_in:
            lines = file_in.readlines()
            lines = lines[2:]
            _file.writelines(lines)
        _file.write('\n')
        for i, bond_pair in zip(range(n_frozen_bonds), bond_pairs_changed):
            _file.write('bond'+str(i)+'(freeze)=B('+str(bond_pair[0]+1)+',')
            _file.write(str(bond_pair[1]+1)+')'+'\n')
        _file.write('\n')

        jobid = os.popen('submit_gaus16 ' + com_file).read()
        jobid = set([int(jobid.split()[-1])])
        out_file = com_file[:-4]+".out"
    return out_file, jobid


def get_frequencies(out_file, n_atoms):
    """
    This function extracts the calculated normal modes from a frequency
    calculation with the corresponding frequencies including the optimized
    geometry
    """
    frequency_list = []
    vibration_matrices = []
    with open(out_file, 'r') as _file:
        line = _file.readline()
        while line:
            if 'Frequencies --' in line:
                data = line.split()
                frequency_list.extend(data[-3:])
                vib1 = np.zeros((n_atoms, 3))
                vib2 = np.zeros((n_atoms, 3))
                vib3 = np.zeros((n_atoms, 3))
                for _ in range(5):
                    line = _file.readline()
                for i in range(n_atoms):
                    data = list(map(np.float, line.split()[2:]))
                    vib1[i, :] = np.array(data[:3])
                    vib2[i, :] = np.array(data[3:6])
                    vib3[i, :] = np.array(data[6:])
                    line = _file.readline()
                vibration_matrices.extend([vib1, vib2, vib3])
            if 'Coordinates (Angstroms)' in line:
                coordinates = np.zeros((n_atoms, 3))
                for _ in range(3):
                    line = _file.readline()
                for i in range(n_atoms):
                    data = list(map(np.float, line.split()[3:]))
                    coordinates[i, :] = data
                    line = _file.readline()

            line = _file.readline()
    frequency_list = list(map(np.float, frequency_list))
    return frequency_list, vibration_matrices, coordinates

def check_imaginary_frequencies(frequency_list, vibration_matrices,
                                bond_breaking_pairs, coordinates, n_atoms):
    """
    This function checks imaginary frequencies by projecting them onto each of
    the atom pairs that have bonds being formed or broken.
    """
    bond_matrices = []
    n_imag_freqs = sum(n < 0 for n in frequency_list)
    print("Number of imaginary frequencies = "+str(n_imag_freqs))
    if n_imag_freqs == 0:
        print("No imaginary Frequencies")
        lowest_freq_active = None
    else:
        for pair in bond_breaking_pairs:
            atom_1_coordinates = coordinates[pair[0], :]
            atom_2_coordinates = coordinates[pair[1], :]
            transition_direction = atom_2_coordinates - atom_1_coordinates
            transition_matrix = np.zeros((n_atoms, 3))
            transition_matrix[pair[0], :] = transition_direction
            transition_matrix[pair[1], :] = -transition_direction
            bond_matrices.append(transition_matrix)

        for i in range(n_imag_freqs):
            if i == 0:
                lowest_freq_active = 0
            print("transition: "+str(i+1))
            frequency_vector = np.ravel(vibration_matrices[i])
            for count, bond_matrix in enumerate(bond_matrices):
                transition_vector = np.ravel(bond_matrix)
                overlap = \
                (transition_vector/np.linalg.norm(transition_vector)) @ frequency_vector
                print(bond_breaking_pairs[count], overlap)
                if abs(overlap) > 0.33:
                    print("Vibration along the bond")
                    if i == 0:
                        lowest_freq_active += 1
                else:
                    print("Vibration not along bond")
        print("Lowest imaginary frequency active along: \
              "+str(lowest_freq_active)+" bonds")

    return n_imag_freqs, lowest_freq_active


def extract_optimized_structure(out_file, n_atoms, atom_labels):
    """
    After waiting for the constrained optimization to finish, the
    resulting structure from the constrained optimization is
    extracted and saved as .xyz file ready for TS optimization.
    """
    optimized_xyz_file = out_file[:-4]+".xyz"
    optimized_energy = None
    with open(out_file, 'r') as ofile:
        line = ofile.readline()
        while line:
            if 'SCF Done:' in line:
                optimized_energy = line.split()[4]
            if 'Standard orientation' in line or 'Input orientation' in line:
                coordinates = np.zeros((n_atoms, 3))
                for i in range(5):
                    line = ofile.readline()
                for i in range(n_atoms):
                    coordinates[i, :] = np.array(line.split()[-3:])
                    line = ofile.readline()
            line = ofile.readline()
    with open(optimized_xyz_file, 'w') as _file:
        _file.write(str(n_atoms)+'\n\n')
        for i in range(n_atoms):
            _file.write(atom_labels[i])
            for j in range(3):
                _file.write(' '+"{:.5f}".format(coordinates[i, j]))
            _file.write('\n')

    print("optimized energy ("+out_file+") = "+str(optimized_energy))

    return optimized_xyz_file, optimized_energy

def do_freq_calculation(xyz_file, method):
    """
    This script sets up a Gaussian frequency calculation using the structure
    specified in xyz_file.
    submits the .com file created using the bash submit script 'submit_gaus16'
    to slurm which returns a jobid used to monitor the job.
    """
    com_file = xyz_file[:-4]+'_freq.com'
    with open(com_file, 'w') as _file:
        _file.write('%mem=16GB'+'\n')
        _file.write('%nprocshared=4'+'\n')
        _file.write('#freq '+str(method)+' ')
        _file.write('scf=(maxcycles=1024)'+'\n\n')
        _file.write('something title'+'\n\n')
        _file.write('0 1'+'\n') #hardcoded to neutral singlet - change if needed
        with open(xyz_file, 'r') as file_in:
            lines = file_in.readlines()
            lines = lines[2:]
            _file.writelines(lines)
        _file.write('\n')

    jobid = os.popen('submit_gaus16 ' + com_file).read()
    jobid = set([int(jobid.split()[-1])])

    out_file = com_file[:-4]+".out"
    return out_file, jobid


def do_gaussian_ts_calculation(ts_guess_xyz, method):
    """
    This script sets up a Gaussian TS calculation using the constrained
    optimized structure.
    submits the .com file created using the bash submit script 'submit_gaus16'
    to slurm which returns a jobid used to monitor the job.
    """
    com_file = ts_guess_xyz[:-4]+'_ts.com'
    with open(com_file, 'w') as _file:
        _file.write('%mem=16GB'+'\n')
        _file.write('%nprocshared=4'+'\n')
        _file.write('#opt=(calcall, ts, noeigen) '+str(method)+' ')
        _file.write('scf=(maxcycles=1024)'+'\n\n')
        _file.write('something title'+'\n\n')
        _file.write('0 1'+'\n')
        with open(ts_guess_xyz, 'r') as file_in:
            lines = file_in.readlines()
            lines = lines[2:]
            _file.writelines(lines)
        _file.write('\n')

    jobid = os.popen('submit_gaus16 ' + com_file).read()
    jobid = set([int(jobid.split()[-1])])

    out_file = com_file[:-4]+".out"
    return out_file, jobid

def fast_ts_check(ts_file, vibrations, coordinates, atom_labels, method):
    """
    This displaces the calculated TS structure along the imaginary frequency of
    the calculated TS structure in each direction and runs an optimization
    calculation on these two structures.
    """
    file_names = [ts_file[:-4]+'_fast_forward.com', ts_file[:-4]+'_fast_reverse.com']
    forward_coordinates = coordinates+0.2*vibrations[0]
    reverse_coordinates = coordinates-0.2*vibrations[0]
    for file_name, coord in zip(file_names, [forward_coordinates, \
            reverse_coordinates]):
        with open(file_name, 'w') as _file:
            _file.write('%mem=16GB'+'\n')
            _file.write('%nprocshared=4'+'\n')
            _file.write('#opt freq '+str(method)+' ')
            _file.write('scf=(maxcycles=1024)'+'\n\n')
            _file.write('something title'+'\n\n')
            _file.write('0 1'+'\n') #hardcoded to neutral singlet - change if needed
            for count, atom_label in enumerate(atom_labels):
                _file.write(atom_label)
                for j in range(3):
                    _file.write(' '+"{:.5f}".format(coord[count, j]))
                _file.write('\n')
            _file.write('\n')

    job_ids = set()
    for _file in file_names:
        jobid = os.popen('submit_gaus16 '+_file).read()
        jobid = int(jobid.split()[-1])
        job_ids.add(jobid)
    out_files = [com_file[:-4]+".out" for com_file in file_names]
    return out_files, job_ids





def make_irc(ts_test_file, method, direction):
    """
    After TS calculation finishes: do an IRC on the optimized TS structure in
    the specified direction (forward or reverse) to see
    if it creates correct reactant and product structures.
    submits the .com file created using the bash submit script 'submit_gaus16'
    to slurm which returns a jobid used to monitor the job.
    """
    irc_file = ts_test_file[:-4]+'_'+direction+'_irc.com'
    with open(irc_file, 'w') as _file:
        _file.write('%mem=16GB'+'\n')
        _file.write('%nprocshared=4'+'\n')
        _file.write('#irc=('+direction+' calcfc, maxpoint=100, stepsize=5) '+str(method)+' ')
        _file.write('scf=(maxcycles=1024)'+'\n\n')
        _file.write('something title'+'\n\n')
        _file.write('0 1'+'\n') #hardcoded to neutral singlet - change if needed
        with open(ts_test_file, 'r') as file_in:
            lines = file_in.readlines()
            lines = lines[2:]
            _file.writelines(lines)
        _file.write('\n')

    jobid = os.popen('submit_gaus16 ' + irc_file).read()
    jobid = set([int(jobid.split()[-1])])

    out_file = irc_file[:-4]+".out"
    return out_file, jobid


def optimize_endpoint(xyz_file, n_atoms, atom_labels, method):
    """
    After the IRC finishes: compare the resulting reactant and product
    structure with the original reactant and product structures
    Use xyz2mol in order to compare the smiles
    submits the .com file created using the bash submit script 'submit_gaus16'
    to slurm which returns a jobid used to monitor the job.

    """
    com_file = xyz_file[:-4]+'_opt.com'
    with open(com_file, 'w') as _file:
        _file.write('%mem=16GB'+'\n')
        _file.write('%nprocshared=4'+'\n')
        _file.write('# opt freq '+str(method)+' ')
        _file.write('scf=(maxcycles=1024)'+'\n\n')
        _file.write('something title'+'\n\n')
        _file.write('0 1'+'\n') #hardcoded as neutral singlet - change if needed
        with open(xyz_file, 'r') as file_in:
            lines = file_in.readlines()
            lines = lines[2:]
            _file.writelines(lines)
        _file.write('\n')

    jobid = os.popen('submit_gaus16 ' + com_file).read()
    jobid = set([int(jobid.split()[-1])])

    out_file = com_file[:-4]+".out"
    return out_file, jobid


def is_ts_correct(reactant_xyz, product_xyz, irc_start_out, irc_end_out,
                  n_atoms, atom_labels):
    """
    This function compares the input smiles with the smiles of the endpoints of
    the IRC.
    """

    set_input = []
    set_irc = []

    for _file in [reactant_xyz, product_xyz]:
        acm = get_adjacency_matrix(_file)
        set_input.append(acm)

    for _file in [irc_start_out, irc_end_out]:
        xyz_file, _ = extract_optimized_structure(_file, n_atoms, atom_labels)
        acm = get_adjacency_matrix(xyz_file)
        set_irc.append(acm)

    print(set_input[0]-set_irc[0])
    print(set_input[1]-set_irc[1])
    print(set_input[1]-set_irc[0])
    print(set_input[0]-set_irc[1])
    if not np.any(set_input[0] - set_irc[0]) and \
    not np.any(set_input[1]-set_irc[1]):
        print("woopwoop TS found")
        ts_found = True
    elif not np.any(set_input[1] - set_irc[0]) and \
    not np.any(set_input[0]-set_irc[1]):
        print("woopwoop TS found")
        ts_found = True
    else:
        print("damn")
        ts_found = False

    return ts_found



def submit_xtb_path_and_test(reactant_file, product_file, kpush, kpull, alp):
    """
    This functions submits an xtb path calculation for the given reaction_file
    and product_file with the given kpull and kpush values. It then waits for
    the calculation to finish and checks whether either of the three paths
    resulted in a completed reaction and returns this information as a list of
    boolean statements.
    """
    job_ids = submit_xtb_path_calculation(reactant_file, product_file, kpush,
                                          kpull, alp)
    wait_for_jobs_to_finish(job_ids)
    out_file = reactant_file[:-4]+'_'+product_file[:-4]+'/' \
            + reactant_file[:-4]+'_'+product_file[:-4]+'.out'
    reactions_completed_boolean = check_if_reaction_complete(out_file)

    return reactions_completed_boolean


def try_xtb_path(reactant_file, product_file, kpull, kpush, alpha):
    """
    Trying an xtb path and checks is reactant completed
    """
    reactions_completed_boolean = submit_xtb_path_and_test(reactant_file, product_file,
                                                           kpull, kpush, alpha)
    print(reactions_completed_boolean)
    if True in reactions_completed_boolean:
        path_file = reactant_file[:-4]+'_'+product_file[:-4]+'/xtbpath_' \
                    +str(reactions_completed_boolean.index(True)+1)+'.xyz'
        out_file = reactant_file[:-4]+'_'+product_file[:-4] \
                   +'/'+reactant_file[:-4]+'_'+product_file[:-4]+'.out'
        return [path_file, out_file, reactions_completed_boolean.index(True)]

    print("trying again")
    path_file = reactant_file[:-4]+'_'+product_file[:-4]+'/xtbpath_1.xyz'
    reactant_file = reactant_file[:-4]+'_rel.xyz'
    get_relaxed_xtb_structure(path_file, reactant_file)
    return [reactant_file, product_file]



def find_xtb_path(reactant_file, product_file):
    """
    Find an xtb path combining the reactant and product file
    """
    kpush_list = [-0.02, -0.02, -0.02, -0.03, -0.03]
    alp_list = [0.6, 0.3, 0.3, 0.6, 0.6]
    output_list = try_xtb_path(reactant_file, product_file, 0.008,  #change initial push to lower
                               kpush_list[0], alp_list[0])
    i = 1
    print(len(output_list))
    while len(output_list) != 3:
        reactant_file = output_list[1]
        product_file = output_list[0]
        output_list = try_xtb_path(reactant_file, product_file, 0.01,
                                   kpush_list[i], alp_list[i])
        i += 1
        print(i)
        if i == 6:
            print("xtb path not found")
            sys.exit()

    return tuple(output_list)


def xtb_path_parameter(n_path, out_file, _dict):
    """
    This function extracts the kpull, kpush and alp values used in the
    succesful xtb path
    """
    push_pull_list = []
    with open(out_file, 'r') as _file:
        line = _file.readline()
        while line:
            if 'Gaussian width (1/Bohr)' in line:
                alp = line.split()[4]
                _dict.update({"ALP [1/a0]" : alp})
            if 'actual k push/pull :' in line:
                push_pull_list.append(line)
            line = _file.readline()
    k_push_pull = push_pull_list[n_path].split()
    _dict.update({"k_push": np.float(k_push_pull[4]), "k_pull" :
                  np.float(k_push_pull[5])})
    return _dict



def find_dft_max_structure(method):
    """
    Once an xtb path has been calculated the xTB guess is refined with an
    interpolation around the xtb max using the method specified in 'method'.
    The maximum of this path is used in the search for a TS.
    """
    files_list = ['max_structure-1.xyz', 'max_structure.xyz',
                  'max_structure+1.xyz']
    atom_numbers, coordinates_list, n_atoms =\
    get_coordinates(files_list)
    n_points = 10
    make_sp_interpolation(atom_numbers, coordinates_list, n_atoms, n_points)
    job_ids = submit_gaussian_sp("path", method)
    wait_for_jobs_to_finish(job_ids)
    sp_max_file, sp_dft_energies = find_dft_max_from_sp_interpolation("path")
    return sp_max_file, sp_dft_energies, n_atoms, atom_numbers

def find_ts_structure(ts_guess_xyz_path, reactant_file, product_file, \
                      bond_pairs_changed, method, n_atoms, atom_numbers):
    """
    Once an xtb path has been calculated the xtb guess is refined and used as
    starting guess in DFT TS berny optimization and tested through an IRC.
    """
    _dict = {}

    #check bond activity requirement of ts guess structure
    bond_activity = check_activity_of_bonds(ts_guess_xyz_path, bond_pairs_changed)
    print(bond_activity)
    _dict.update({"Bond active in guess": any(bond_activity)})
    _, ts_guess_xyz = os.path.split(ts_guess_xyz_path)
    try:
        shutil.copyfile(ts_guess_xyz_path, "ts_test/"+ts_guess_xyz)
    except shutil.SameFileError:
        print("file already there")
    #create directory to do DFT TS calculations
    os.chdir("ts_test")

    #Calculate vibratins of TS guess and do TS optimization
    freq_outfile, job_id1 = do_freq_calculation(ts_guess_xyz,
                                                method)
    ts_out_file, job_id2 = \
    do_gaussian_ts_calculation(ts_guess_xyz, method)
    job_ids = job_id1|job_id2
    wait_for_jobs_to_finish(job_ids)

    #analyze vibrations in TS guess
    frequencies, vibrations, coordinates = get_frequencies(freq_outfile, n_atoms)
    n_imag_freqs_guess, lowest_vibration_active_guess = \
    check_imaginary_frequencies(frequencies, vibrations, bond_pairs_changed,
                                coordinates, n_atoms)
    _dict.update({"N_imag_guess": n_imag_freqs_guess})
    _dict.update({"# of bonds along lowest vibration in guess": \
        lowest_vibration_active_guess})

    #analyze energy and vibrations in TS structure
    ts_xyz_file, ts_energy = extract_optimized_structure(ts_out_file, n_atoms,
                                                         atom_numbers[0])
    _dict.update({"TS Energy [Hartree]": ts_energy})
    frequencies, vibrations, coordinates = get_frequencies(ts_out_file, n_atoms)

    n_imag_freqs_ts, lowest_vibration_active_ts = check_imaginary_frequencies(frequencies,
                                                                              vibrations,
                                                                              bond_pairs_changed,
                                                                              coordinates, n_atoms)
    _dict.update({"N_imag_ts": n_imag_freqs_ts})
    _dict.update({"# of bonds along lowest vibration in TS": \
        lowest_vibration_active_ts})

    bond_activity = check_activity_of_bonds(ts_xyz_file, bond_pairs_changed)
    print(bond_activity)
    _dict.update({"Bond active in ts": any(bond_activity)})

    #Do a fast check of TS (step once in each direction if vibration and optimize)
    fast_check_files, job_ids = fast_ts_check(ts_xyz_file, vibrations, coordinates,
                                              atom_numbers[0], method)

    #Calculate IRC in both directions
    irc_for_out_file, job_id = make_irc(ts_xyz_file, method, 'forward')
    irc_rev_out_file, job_id2 = make_irc(ts_xyz_file, method, 'reverse')
    job_ids = job_ids|job_id|job_id2
    wait_for_jobs_to_finish(job_ids)

    #check the "fast test" structures 
    print("checking fast ts check:")
    ts_found = is_ts_correct('../'+reactant_file, '../'+product_file, fast_check_files[0],
                             fast_check_files[1], n_atoms, atom_numbers[0])
    _dict.update({"Fast test": ts_found})

    #extract and test the IRC endpoint structures
    irc_for_xyz_file, _ = extract_optimized_structure(irc_for_out_file,
                                                      n_atoms, atom_numbers[0])
    irc_rev_xyz_file, _ = extract_optimized_structure(irc_rev_out_file,
                                                      n_atoms, atom_numbers[0])
    print("Checking IRC endpoints:")
    try:
        ts_found = is_ts_correct('../'+reactant_file, '../'+product_file,
                                 irc_for_out_file, irc_rev_out_file, n_atoms, atom_numbers[0])
    except UnboundLocalError:
        print("something wrong with IRC")
        ts_found = None
    _dict.update({"IRC endpoint test": ts_found})

    #Optimize IRC endpoint structures
    irc_for_opt_out, jobid = optimize_endpoint(irc_for_xyz_file, n_atoms, atom_numbers[0], method)
    irc_rev_opt_out, jobid2 = optimize_endpoint(irc_rev_xyz_file, n_atoms, atom_numbers[0], method)
    jobids = jobid|jobid2
    wait_for_jobs_to_finish(jobids)

    #check optimized IRC enpoint structures
    print("checking real IRC:")
    extract_optimized_structure(irc_for_opt_out, n_atoms, atom_numbers[0])
    extract_optimized_structure(irc_rev_opt_out, n_atoms, atom_numbers[0])
    try:
        ts_found_opt = is_ts_correct('../'+reactant_file, '../'+product_file, \
                irc_for_opt_out, irc_rev_opt_out, n_atoms, atom_numbers[0])
    except UnboundLocalError:
        print("something wrong with IRC")
        ts_found_opt = None
    _dict.update({"IRC check": ts_found})


    return _dict, ts_found_opt, ts_found


if __name__ == '__main__':
    FILE_1 = sys.argv[1] #reactant .xyz file
    FILE_2 = sys.argv[2] #product .xyz file
    SUCCESS_PKL = sys.argv[3] #.pkl file with dataframe to save result if search successful
    FAIL_PKL = sys.argv[4] #.pkl faile with daraframe to save result if search unsuccessful
    METHOD = 'ub3lyp/6-31G(d,p)' #Specify the method for the Gaussian calculations
    LG = RDLogger.logger()
    LG.setLevel(RDLogger.ERROR)


    with open("log_err.txt", 'w') as err:
        with redirect_stderr(err):
            with open("log.txt", 'w') as out:
                with redirect_stdout(out):
                    # create empty dictionary to save results
                    DICT = {}

                    #Get the xTB path for reactant and product
                    PATH_FILE, OUTFILE, N_PATH = find_xtb_path(FILE_1, FILE_2)
                    DICT = xtb_path_parameter(N_PATH, OUTFILE, DICT)

                    #extract path structures and do sp energy calculations
                    XTB_COORDINATES = extract_xtb_structures(PATH_FILE)
                    JOBIDS = submit_xtb_sp("xtb_sp")
                    wait_for_jobs_to_finish(JOBIDS)

                    #find maximum energy along path (xTB)
                    MAX_ENERGY_SP, XTB_SP_ENERGIES_KCAL, _ = \
                            find_xtb_max_from_sp_interpolation("xtb_sp", True)

                    #plot xTB path
                    plot_path(XTB_COORDINATES, XTB_SP_ENERGIES_KCAL, 'indigo',
                              'xTB SP', 'Path length [a$_0$]')
                    plt.savefig('xtb_sp_path.pdf')
                    plt.clf()

                    #Find bonds getting broken or formed during reaction
                    BOND_PAIRS_CHANGED = find_bonds_getting_formed_or_broken(FILE_1,
                                                                             FILE_2)
                    SP_MAX_XYZ, SP_DFT_ENERGIES, N_ATOMS, ATOM_NUMBERS = find_dft_max_structure(METHOD)
                    JOBIDS = submit_xtb_sp("path")
                    wait_for_jobs_to_finish(JOBIDS)
                    MAX_XTB_ENERGY, XTB_SP_ENERGIES_KCAL, MAX_INDEX = find_xtb_max_from_sp_interpolation("path", False)
                    DICT.update({"xTB max [Hartree]" : MAX_XTB_ENERGY})
                    DICT.update({"DFT SP Max [Hartree]": max(SP_DFT_ENERGIES)})
                    #compare max DFT and xTB structures of the interpolation
                    XTB_EQ_DFT_MAX = MAX_INDEX == SP_DFT_ENERGIES.index(max(SP_DFT_ENERGIES))
                    DICT.update({"DFT + xTB index same": XTB_EQ_DFT_MAX})
                    DICT.update({"index dif": \
                        MAX_INDEX-SP_DFT_ENERGIES.index(max(SP_DFT_ENERGIES))})
                    print(DICT)

                    #plot the xTB and DFT interpolations
                    plot_path(np.arange(-10, 11, 1), XTB_SP_ENERGIES_KCAL,
                              'indigo', 'xTB SP', 'Path point')
                    plt.savefig('xtb_interpolation.pdf')
                    DFT_SP_ENERGIES_KCAL = \
                    (np.array(SP_DFT_ENERGIES)-SP_DFT_ENERGIES[0])*627.509
                    plot_path(np.arange(-10, 11, 1), DFT_SP_ENERGIES_KCAL,
                              'orange', "DFT SP", 'Path point')
                    plt.savefig("xtb+dft_sp.pdf")
                    plt.clf()
                    plot_path(np.arange(-10, 11, 1), DFT_SP_ENERGIES_KCAL,
                              'orange', 'DFT SP', 'Path point')
                    plt.savefig('dft_interpolation.pdf')

                    #search for TS and validate with IRC
                    os.mkdir('ts_test')
                    DICT_NOOPT, TS_FOUND1, TS_FOUND2 =\
                        find_ts_structure(SP_MAX_XYZ, FILE_1, FILE_2, BOND_PAIRS_CHANGED,
                                          METHOD, N_ATOMS, ATOM_NUMBERS)
                    os.chdir('../')
                    DICT_NOOPT.update(DICT)
                    print(DICT_NOOPT)

                    #Get reaction index from file name - used as index in the dataframe
                    REACTION_INDEX = int(''.join(filter(lambda i: i.isdigit(), FILE_1)))

                    #save dictionary in a .pkl file
                    DF = pd.DataFrame(DICT_NOOPT, index=[REACTION_INDEX])
                    DF.to_pickle(str(REACTION_INDEX)+'_noopt.pkl')

                    #load the dataframes to save data for successful and failed runs
                    SUCCESS_DATAFRAME = pd.read_pickle('../'+SUCCESS_PKL)
                    FAIL_DATAFRAME = pd.read_pickle('../'+FAIL_PKL)

                    # check if TS found if not do constraind optimization 
                    if TS_FOUND1 or TS_FOUND2:
                        SUCCESS_DATAFRAME = pd.read_pickle('../'+SUCCESS_PKL)
                        SUCCESS_DATAFRAME = SUCCESS_DATAFRAME.append(DF,
                                                                     sort=False)
                        SUCCESS_DATAFRAME.to_pickle('../'+SUCCESS_PKL)

                        #terminate program if TS found
                        sys.exit()

                    #Do constrained optimization on the SP max structure
                    _, SP_MAX_XYZ = os.path.split(SP_MAX_XYZ)
                    os.chdir('ts_test')
                    CONSTRAINED_OPT_OUT, JOB_ID = \
                            make_gaussian_constrained_opt(SP_MAX_XYZ,
                                                          BOND_PAIRS_CHANGED, METHOD)
                    wait_for_jobs_to_finish(JOB_ID)
                    CONSTRAINED_OPT_XYZ, CONSTRAINED_OPT_ENERGY = \
                        extract_optimized_structure(CONSTRAINED_OPT_OUT, N_ATOMS, ATOM_NUMBERS[0])
                    DICT.update({"Constrained opt [Hartree]" : CONSTRAINED_OPT_ENERGY})
                    os.chdir('../')
                    CONSTRAINED_OPT_XYZ = 'ts_test/'+CONSTRAINED_OPT_XYZ

                    #try TS search again on constrained opt save results in new dictionary
                    DICT_OPT, TS_FOUND1, TS_FOUND2 = \
                    find_ts_structure(CONSTRAINED_OPT_XYZ, FILE_1, FILE_2, BOND_PAIRS_CHANGED,
                                      METHOD, N_ATOMS, ATOM_NUMBERS)
                    os.chdir('../')
                    DICT_OPT.update(DICT)
                    print(DICT_OPT)

                    #save dataframe for the constrained optimixation in .pkl
                    DF2 = pd.DataFrame(DICT_OPT, index=[REACTION_INDEX])
                    DF2.to_pickle(str(REACTION_INDEX)+'_opt.pkl')

                    # check again if TS found
                    if TS_FOUND1 or TS_FOUND2:
                        SUCCESS_DATAFRAME = pd.read_pickle('../'+SUCCESS_PKL)
                        SUCCESS_DATAFRAME = SUCCESS_DATAFRAME.append(DF2, sort=False)
                        SUCCESS_DATAFRAME.to_pickle('../'+SUCCESS_PKL)

                        #exit if TS found else save results for the failed attempts
                        sys.exit()

                    #if TS not found save failed attempts for both tries in dataframe
                    FAIL_DATAFRAME = pd.read_pickle('../'+FAIL_PKL)
                    FAIL_DATAFRAME = FAIL_DATAFRAME.append([DF, DF2], sort=False)
                    FAIL_DATAFRAME.to_pickle('../'+FAIL_PKL)
