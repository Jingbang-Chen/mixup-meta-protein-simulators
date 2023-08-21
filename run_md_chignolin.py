import os
from tqdm import tqdm 

from openmm import *
from openmm.app import *
from openmmtools.integrators import VVVRIntegrator


def gen_data(pdb_filename='output.pdb', results_prefix = 'save_trajs', temperature = 300):
    pdb = PDBFile(pdb_filename)

    topology = pdb.topology
    positions = pdb.positions

    forcefield = ForceField('amber99sb.xml', 'tip3p.xml')

    system = forcefield.createSystem(
                topology,
                nonbondedMethod=NoCutoff,
                nonbondedCutoff=1.0*unit.nanometers,
                constraints=HBonds,
                ewaldErrorTolerance=0.0005)

    integrator = VVVRIntegrator(temperature*unit.kelvin, 1.0/unit.picoseconds, 2.0*unit.femtoseconds)
    integrator.setConstraintTolerance(0.00001)

    platform = Platform.getPlatformByName('CUDA')
    platformProperties = {'Precision': 'mixed'}

    simulation = Simulation(topology, system, integrator, platform, platformProperties)
    simulation.context.setPositions(positions)

    simulation.minimizeEnergy()

    simulation.context.setVelocitiesToTemperature(temperature*unit.kelvin)

    if not os.path.exists(results_prefix):
        os.makedirs(results_prefix)

    for i in tqdm(range(1000)):
        simulation.step(100)
        position = simulation.context.getState(getPositions=True).getPositions()
        PDBFile.writeFile(simulation.topology, position, open(f'{results_prefix}/traj_{i}.pdb', 'w'))




