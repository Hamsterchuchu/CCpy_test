import os, sys
from subprocess import call as shl
from collections import OrderedDict
import yaml
from CCpy.Queue.server_profile import get_node_profile
def represent_dictionary_order(self, dict_data):
    return self.represent_mapping('tag:yaml.org,2002:map', dict_data.items())
yaml.add_representer(OrderedDict, represent_dictionary_order)
import warnings
warnings.filterwarnings("ignore")
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# -- version chk
version = sys.version
if version[0] == '3':
    raw_input = input

# -------------------- Config ---------------------#
"""
Up to own HPC system
Only SGE queue system allowed
"""

# -- Queue and nodes settings
try:
    CCpy_SCHEDULER_CONFIG = os.environ['CCpy_SCHEDULER_CONFIG']
except:
    print('''Error while load $CCpy_SCHEDULER_CONFIG file.
Please check the example of scheduler config file at https://github.com/91bsjun/CCpy/tree/master/CCpy/Queue''')
    quit()

queue_info = yaml.load(open(CCpy_SCHEDULER_CONFIG, 'r'))


class JobSubmit:
    def __init__(self, inputfile, queue, n_of_cpu, node=None, init_only=False):
        home = os.getenv("HOME")
        user_queue_config = "%s/.CCpy/queue_config.yaml" % home
        if not os.path.isfile(user_queue_config):
            from pathlib import Path
            MODULE_DIR = Path(__file__).resolve().parent
            default_queue_config = str(MODULE_DIR) + "/queue_config.yaml"
            if ".CCpy" not in os.listdir(home):
                os.mkdir("%s/.CCpy" % home)
            os.system('cp %s %s' % (default_queue_config, user_queue_config))

        if init_only:
            return

        self.inputfile = inputfile
        cpu, mem, q = queue_info[queue][0], queue_info[queue][1], queue_info[queue][2]

        self.cpu = cpu
        self.mem = mem
        self.q = q
        if n_of_cpu:
            self.n_of_cpu = n_of_cpu
        else:
            self.n_of_cpu = cpu
        self.divided = cpu / self.n_of_cpu

        # -- read configs from queue_config.yaml            
        yaml_string = open(user_queue_config, "r").read()
        queue_config = yaml.load(yaml_string)
            
        self.qsub = queue_config['qsub']

        self.python_path = queue_config['python_path']
        self.mpi_run = queue_config['mpi_run']

        self.atk_mpi_run = queue_config['atk_mpi_run']
        self.atk_mpi_run = queue_config['mpi_run']

        vasp_path = queue_config['vasp_path']
        self.vasp_run = "%s %s < /dev/null > vasp.out" % (self.mpi_run, vasp_path)

        self.g09_path = queue_config['g09_path']
        self.atk_path = queue_config['atk_path']
        
        # -- optional: LAMMPS can use its own mpi launch command (e.g. "srun --mpi=pmi2")
        # without affecting mpi_run used by vasp/siesta/atk. Falls back to mpi_run if not set.
        self.lammps_mpirun_path = queue_config.get('lammps_mpi_run', self.mpi_run)
        self.lammps_path = queue_config['lammps_path']
        # -- optional: pre-run environment setup (e.g. conda activate) for custom LAMMPS builds
        # not required in queue_config.yaml; defaults to empty string if not set
        self.lammps_env = queue_config.get('lammps_env', "")

        self.siesta_path = queue_config['siesta_path']


        # -- queue settings
        self.pe_request = "#SBATCH -n %d #core" % self.n_of_cpu
        self.queue_name = "#$ -q %s" % self.q if self.q else ""
        self.node_assign = ""
     
        if self.n_of_cpu > 48:
            use_node = int(self.n_of_cpu / 48)
            self.node_assign = "#SBATCH -N %d #node" % use_node
        else:
            self.node_assign = "#SBATCH -N 1"

        # -- 서버별 노드/파티션 프로파일 (CCpy/Queue/server_profile.py)
        #    cms2 와 node99 는 노드 이름과 파티션 구성이 서로 다르다.
        #    예전에는 두 서버가 이 부분을 각자 직접 고쳐 써서 코드가 갈라져 있었다.
        profile = get_node_profile()
        self.allot_node = ""
        self.partition_name = profile['default_partition']
        if node:
            self.allot_node = "#SBATCH --nodelist=%s" % node
            # 노드 이름 -> 파티션
            for part_name, node_list in profile['node_partitions'].items():
                if node in node_list:
                    self.partition_name = part_name
                    break
            # 코어수만 지정한 경우 (예: '96') -> 특정 노드 지정 없이 파티션 전체
            if node in profile['partition_alias']:
                self.partition_name = profile['partition_alias'][node]
                self.allot_node = ""

    def gaussian(self, ):
        inputfile = self.inputfile
        cpu, mem, q = self.n_of_cpu, self.mem, self.q
        d = self.divided

        mem = int(mem / d)

        f = open(inputfile, "r")
        lines = f.readlines()
        f.close()

        f = open(inputfile, "w")
        for line in lines:
            if "%nproc=" in line:
                f.write("%nproc=" + str(cpu) + "\n")
            elif "%mem=" in line:
                f.write("%mem=" + str(mem) + "Gb\n")
            else:
                f.write(line)
        f.close()

        jobname = "G" + inputfile.replace(".com", "")
        jobname = jobname.replace(".", "_").replace("-", "_")

        mpi = '''#!/bin/sh
#SBATCH -J %s         # jobname
#SBATCH -p %s     # partition name
%s
# n of nodes
%s
# n of cpu
%s

#SBATCH -o %%x.o%%j
#SBATCH -e %%x.e%%j

export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi.so      # Do not change here!!

cd $SLURM_SUBMIT_DIR
%s %s

 ''' % (jobname, self.partition_name, self.allot_node, self.pe_request, self.node_assign, self.g09_path, inputfile)

        f = open("mpi.sh", "w")
        f.write(mpi)
        f.close()

        shl(self.qsub + " mpi.sh", shell=True)
        shl("rm -rf ./mpi.sh", shell=True)

    def gaussian_batch(self, input_files):
        cpu, mem, q = self.n_of_cpu, self.mem, self.q
        d = self.divided

        mem = int(mem / d)

        for inputfile in input_files:
            f = open(inputfile, "r")
            lines = f.readlines()
            f.close()

            f = open(inputfile, "w")
            for line in lines:
                if "%nproc=" in line:
                    f.write("%nproc=" + str(cpu) + "\n")
                elif "%mem=" in line:
                    f.write("%mem=" + str(mem) + "Gb\n")
                else:
                    f.write(line)
            f.close()

        jobname = raw_input("Jobname for this job \n: ")
        runs = ""
        for each_input in input_files:
            runs += "%s %s\nsleep 10\n" % (self.g09_path, each_input)

        mpi = '''#!/bin/sh
#SBATCH -J %s         # jobname
#SBATCH -p %s     # partition name
%s
# n of nodes
%s
# n of cpu
%s

#SBATCH -o %%x.o%%j
#SBATCH -e %%x.e%%j

export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi.so      # Do not change here!!

cd $SLURM_SUBMIT_DIR
%s

 ''' % (jobname, self.partition_name, self.allot_node, self.pe_request, self.node_assign, runs)

        f = open("mpi.sh", "w")
        f.write(mpi)
        f.close()

        shl(self.qsub + " mpi.sh", shell=True)
        shl("rm -rf ./mpi.sh", shell=True)


    def vasp(self, band=False, dirpath=None, loop=False):
        inputfile = self.inputfile

        vasp_run = self.vasp_run
        # -- Band calculation after previous calculation
        if band:
            jobname = "VB" + inputfile
        elif loop:
            from pathlib import Path
            MODULE_DIR = Path(__file__).resolve().parent
            loop_opt_script = str(MODULE_DIR) + "/../Package/VASPOptLoop.py"
            os.system('cp %s ./.VASPOptLoop.py' % loop_opt_script)
            script_filename = ".VASPOptLoop.py"
            script_path = os.getcwd() + "/" + script_filename
            # self.vasp_run = "%s %s\nrm %s" % (self.python_path, script_path, script_path)
            self.vasp_run = "%s %s" % (self.python_path, script_path)
            jobname = "VL" + inputfile
        else:
            jobname = "V" + inputfile
        jobname = jobname.replace(".", "_").replace("-", "_").replace("/", "_")
        mpi = '''#!/bin/sh
#SBATCH -J %s         # jobname
#SBATCH -p %s     # partition name
%s
# n of nodes
%s
# n of cpu
%s

#SBATCH -o %%x.o%%j
#SBATCH -e %%x.e%%j

export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi.so      # Do not change here!!

cd %s
%s
touch vasp.done

 ''' % (jobname, self.partition_name, self.allot_node, self.pe_request, self.node_assign, dirpath, vasp_run)

        pwd = os.getcwd()
        os.chdir(dirpath)
        if 'vasp.done' in os.listdir():
            os.remove('vasp.done')
        f = open("mpi.sh", "w")
        f.write(mpi)
        f.close()
        shl(self.qsub + " mpi.sh", shell=True)
#        shl("rm -rf ./mpi.sh", shell=True)
        os.chdir(pwd)

    def vasp_batch(self, dirs=None, scratch=False, loop=False, jobname=None):
        """
        Run multiple VASP jobs in a single queue
        """
        if not jobname:
            jobname = raw_input("Jobname for this job \n: ")

        runs = ""
        script_path = None

        vasp_run = self.vasp_run
        pwd = os.getcwd()
        if loop:
            from CCpy.Package.VASPOptLoopQueScript import VASPOptLoopQueScriptString
            script_string = VASPOptLoopQueScriptString()
            script_filename = ".VASPOptLoop.py"
            f = open(script_filename, "w")
            f.write(script_string)
            f.close()
            script_path = os.getcwd() + "/" + script_filename
            each_run = "%s %s\ntouch vasp.done\nsleep 30\n" % (self.python_path, script_path)
        else:
            each_run = "%s\ntouch vasp.done\nsleep 30\n" % vasp_run
        for d in dirs:
            # if use scratch, copy input to /scratch/vasp and run job in that dir,
            # when finished, copy to original working directory
            # scratch is recommended when perform small jobs
            os.chdir(d)
            if 'vasp.done' in os.listdir():
                os.remove('vasp.done')
            os.chdir(pwd)
            if scratch:
                dir_path = "/scratch/vasp" + d
                runs += "mkdir -p " + dir_path + "\n"  # make dir under /scratch/vasp
                runs += "cp " + d + "/* " + dir_path + "\n"  # copy original to /scratch/vasp
                runs += "cd " + dir_path + "\n"  # chg dir to /scratch/vasp
                runs += each_run + "\n"  # run vasp
                runs += "cp " + dir_path + "/* " + d + "\n"  # copy finished job to original dir
                runs += "rm -rf " + dir_path + "\n\n"  # remove finished job under /scratch/vasp
            # change dir to each input and run 'each_run'
            else:
                runs += "cd " + d + "\n"
                runs += each_run
        # if loop:
        #    runs += "rm %s" % script_path
        mpi = '''#!/bin/sh
#SBATCH -J %s         # jobname
#SBATCH -p %s     # partition name
%s
# n of nodes
%s
# n of cpu
%s

#SBATCH -o %%x.o%%j
#SBATCH -e %%x.e%%j

export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi.so      # Do not change here!!

%s
         ''' % (jobname, self.partition_name, self.allot_node, self.pe_request, self.node_assign, runs)

        pwd = os.getcwd()
        f = open("mpi.sh", "w")
        f.write(mpi)
        f.close()
        shl(self.qsub + " mpi.sh", shell=True)
        shl("rm -rf ./mpi.sh", shell=True)

    def qchem(self):
        inputfile = self.inputfile
        outputfile = inputfile.replace(".in", ".out")

        jobname = "Q" + inputfile.replace(".in", "")
        jobname = jobname.replace(".", "_").replace("-", "_")

        mpi = '''#!/bin/csh
# Job name 
#$ -N %s

# pe request
%s

# queue name
%s

# node
%s

#$ -V
#$ -cwd

set  MPI_HOME=/opt/mpi/intel-parallel-studio2013sp1/openmpi-1.6.5
set  MPI_EXEC=%s

setenv QCSCRATCH /scratch
setenv QCAUX /opt/QChem4.2/qcaux
source /opt/QChem4.2/qcenv.csh

cd $SGE_O_WORKDIR

qchem %s %s

''' % (jobname, self.pe_request, self.queue_name, self.node_assign, self.mpi_run, inputfile, outputfile)

        f = open("mpi.sh", "w")
        f.write(mpi)
        f.close()

        shl(self.qsub + " mpi.sh", shell=True)
        shl("rm -rf ./mpi.sh", shell=True)

    def ATK(self, atk_version="atk2017"):
        inputfile = self.inputfile

        jobname = "A" + inputfile.replace(".py", "")
        jobname = jobname.replace(".", "_").replace("-", "_")
        outputfile = inputfile.replace(".py", ".out")
        now_path = os.getcwd()
        mpi = '''#!/bin/sh
#SBATCH -J %s         # jobname
#SBATCH -p 48core     # partition name 
%s
# n of cpu
%s                                                                                                                                                                                                                                                           
# n of nodes
%s

#SBATCH -o %s.o%%j 
#SBATCH -e %s.e%%j

export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi.so      # Do not change here!! 

cd %s 

export SNPSLMD_LICENSE_FILE='27020@10.0.0.100'
#export QUANTUM_LICENSE_PATH='6200@166.104.249.199'

%s %s %s > %s

''' % (jobname, self.allot_node, self.pe_request, self.node_assign, jobname, jobname, now_path, self.atk_mpi_run, self.atk_path,
       inputfile, outputfile)

        f = open("mpi.sh", "w")
        f.write(mpi)
        f.close()

        shl(self.qsub + " mpi.sh", shell=True)
        shl("rm -rf ./mpi.sh", shell=True)

    def atat(self):
        dirname = os.getcwd()
        dirname = dirname.split("/")[-1]

        inputfile = self.inputfile

        if "/" in inputfile and "p+" in inputfile:
            jobname = "AT_" + inputfile.split("/")[-1]
        else:
            jobname = "AT_" + dirname + "_" + inputfile
        jobname = jobname.replace(".", "_").replace("-", "_").replace("+", "_")

        os.chdir(inputfile)
        mpi = '''#!/bin/csh
# Job name 
#$ -N %s

# pe request
%s

# queue name
%s

# node
%s

#$ -V
#$ -cwd

echo "Got $NSLOTS slots."
cat $TMPDIR/machines

cd $SGE_O_WORKDIR

runstruct_vasp -ng mpirun -np %d
rm wait
 ''' % (jobname, self.pe_request, self.queue_name, self.node_assign, self.n_of_cpu)

        f = open("mpi.sh", "w")
        f.write(mpi)
        f.close()

        shl(self.qsub + " mpi.sh", shell=True)
        shl("rm -rf ./mpi.sh", shell=True)

    # -- To show SGE queue system that " I'm running now "
    def pbs_runner(self):
        inputfile = self.inputfile

        jobname = inputfile.replace(".py", "")

        mpi = '''#!/bin/csh
# Job name 
#$ -N %s

# pe request
%s

# queue name
%s

# node
%s

#$ -V
#$ -cwd

cd $SGE_O_WORKDIR

python %s

    ''' % (jobname, self.pe_request, self.queue_name, self.node_assign, inputfile)

        f = open("mpi.sh", "w")
        f.write(mpi)
        f.close()

        shl(self.qsub + " mpi.sh", shell=True)
        shl("rm -rf ./mpi.sh", shell=True)

    def lammps(self):
        inputfile = self.inputfile
        outputfile = inputfile.replace("in.", "out.")

        jobname = "L" + inputfile.replace("in.", "")
        jobname = jobname.replace(".", "_").replace("-", "_")

        mpi = '''#!/bin/sh
#SBATCH -J %s         # jobname
#SBATCH -p %s     # partition name
%s
# n of nodes
%s
# n of cpu
%s

#SBATCH -o %%x.o%%j
#SBATCH -e %%x.e%%j

export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi.so      # Do not change here!!

cd $SLURM_SUBMIT_DIR
%s
%s %s < %s | tee %s

    ''' % (jobname, self.partition_name, self.allot_node, self.pe_request, self.node_assign,
           self.lammps_env, self.lammps_mpirun_path, self.lammps_path, inputfile, outputfile)

        f = open("mpi.sh", "w")
        f.write(mpi)
        f.close()

        shl(self.qsub + " mpi.sh", shell=True)
        shl("rm -rf ./mpi.sh", shell=True)

    def lammps_batch(self, inputs=None, jobname=None):
        """
        Run multiple LAMMPS jobs sequentially in a single queue submission
        """
        if not jobname:
            jobname = raw_input("Jobname for this job \n: ")
        jobname = jobname.replace(".", "_").replace("-", "_")

        runs = ""
        for each_input in inputs:
            outputfile = each_input.replace("in.", "out.")
            runs += "%s %s < %s | tee %s\n" % (self.lammps_mpirun_path, self.lammps_path, each_input, outputfile)
            runs += "sleep 10\n\n"

        mpi = '''#!/bin/sh
#SBATCH -J %s         # jobname
#SBATCH -p %s     # partition name
%s
# n of nodes
%s
# n of cpu
%s

#SBATCH -o %%x.o%%j
#SBATCH -e %%x.e%%j

export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi.so      # Do not change here!!

cd $SLURM_SUBMIT_DIR
%s
%s
        ''' % (jobname, self.partition_name, self.allot_node, self.pe_request, self.node_assign,
               self.lammps_env, runs)

        f = open("mpi.sh", "w")
        f.write(mpi)
        f.close()

        shl(self.qsub + " mpi.sh", shell=True)
        shl("rm -rf ./mpi.sh", shell=True)

    def AIMD_NVT_Loop(self, structure_filename=None, temp=None, specie="Li", screen='no_screen', max_step=250):
        # -- load loop queue script
        from CCpy.Package.Diffusion.NVTLoopQueScript import NVTLoopQueScriptString
        script_string = NVTLoopQueScriptString()
        script_filename = ".AIMDLoop.py"
        f = open(script_filename, "w")
        f.write(script_string)
        f.close()

        jobname = "NVT%s_%dK" % (structure_filename.replace(".cif", ""), temp)

        mpi = '''#!/bin/sh
#SBATCH -J %s         # jobname
#SBATCH -p %s     # partition name
%s
# n of nodes
%s
# n of cpu
%s

#SBATCH -o %%x.o%%j
#SBATCH -e %%x.e%%j

export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi.so      # Do not change here!!

%s %s %s %s %s %s %s
''' % (jobname, self.partition_name, self.allot_node, self.pe_request, self.node_assign, self.python_path,
       script_filename, structure_filename, temp, specie, screen, max_step)

        f = open("mpi.sh", "w")
        f.write(mpi)
        f.close()

        shl(self.qsub + " mpi.sh", shell=True)
#        shl("rm -rf ./mpi.sh", shell=True)

    def AIMD_NVT_Loop_batch(self, structure_files=None, temp=None, specie="Li", screen='no_screen', max_step=250):
        # -- load loop queue script
        from CCpy.Package.Diffusion.NVTLoopQueScript import NVTLoopQueScriptString
        script_string = NVTLoopQueScriptString()
        script_filename = ".AIMDLoop.py"
        f = open(script_filename, "w")
        f.write(script_string)
        f.close()

        jobname = input("Job name: ")

        runs = ""
        pwd = os.getcwd()
        if 'structures' not in os.listdir('./'):
            os.mkdir('structures')
        for structure_filename in structure_files:
            dirname = structure_filename.replace(".cif", "")
            runs += "cp %s structures; mkdir %s; mv %s %s; cp %s %s; cd %s\n" % (structure_filename, dirname, structure_filename, dirname, script_filename, dirname, dirname)
            runs += "%s %s %s %s %s %s %s \n\n" % (self.python_path, script_filename, structure_filename, temp, specie, screen, max_step)
            runs += "cd %s \n" % pwd

        mpi = '''#!/bin/csh
# Job name 
#$ -N %s

# pe request
%s

# queue name
%s

# node
%s

#$ -V
#$ -cwd


%s
''' % (jobname, self.pe_request, self.queue_name, self.node_assign, runs)

        f = open("mpi.sh", "w")
        f.write(mpi)
        f.close()

        shl(self.qsub + " mpi.sh", shell=True)
        shl("rm -rf ./mpi.sh", shell=True)

    def casm_run(self):
        jobname = raw_input("Job name: ")

        mpi = '''#!/bin/csh
# Job name 
#$ -N %s

# pe request
%s

# queue name
%s

# node
%s

#$ -V
#$ -cwd

casm-calc --run

        ''' % (jobname, self.pe_request, self.queue_name, self.node_assign)

        f = open("mpi.sh", "w")
        f.write(mpi)
        f.close()

        shl(self.qsub + " mpi.sh", shell=True)
        shl("rm -rf ./mpi.sh", shell=True)

    def siesta(self):
        input_filename = self.inputfile.split("/")[-1]
        dir_path = self.inputfile.replace(input_filename, "")
        jobname = "S" + input_filename.replace(".fdf", "")
        jobname = jobname.replace(".", "_").replace("-", "_")
        mpi = '''#!/bin/sh
#SBATCH -J %s         # jobname
#SBATCH -p %s     # partition name
%s
# n of nodes
%s
# n of cpu
%s

#SBATCH -o %%x.o%%j
#SBATCH -e %%x.e%%j

export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi.so      # Do not change here!!

cd %s
%s %s < %s > siesta.out

        ''' % (jobname, self.partition_name, self.allot_node, self.pe_request, self.node_assign, dir_path, self.mpi_run, self.siesta_path, input_filename)

        f = open("mpi.sh", "w")
        f.write(mpi)
        f.close()

        shl(self.qsub + " mpi.sh", shell=True)
        shl("rm -rf ./mpi.sh", shell=True)

    def siesta_batch(self, inputs=None, jobname=None):
        """
        Run multiple SIESTA jobs sequentially in a single queue submission
        """
        if not jobname:
            jobname = raw_input("Jobname for this job \n: ")
        jobname = jobname.replace(".", "_").replace("-", "_")

        pwd = os.getcwd()
        runs = ""
        for each_input in inputs:
            input_filename = each_input.split("/")[-1]
            dir_path = each_input.replace(input_filename, "")
            if dir_path == "":
                dir_path = "./"
            runs += "cd %s\n" % pwd          # go back to the submission directory first
            runs += "cd %s\n" % dir_path     # then into this job's directory (works for relative or absolute dir_path)
            runs += "%s %s < %s > siesta.out\n" % (self.mpi_run, self.siesta_path, input_filename)
            runs += "touch siesta.done\n"
            runs += "sleep 10\n\n"

        mpi = '''#!/bin/sh
#SBATCH -J %s         # jobname
#SBATCH -p %s     # partition name
%s
# n of nodes
%s
# n of cpu
%s

#SBATCH -o %%x.o%%j
#SBATCH -e %%x.e%%j

export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi.so      # Do not change here!!

%s
        ''' % (jobname, self.partition_name, self.allot_node, self.pe_request, self.node_assign, runs)

        f = open("mpi.sh", "w")
        f.write(mpi)
        f.close()

        shl(self.qsub + " mpi.sh", shell=True)
        shl("rm -rf ./mpi.sh", shell=True)

    def siesta_AIMD_NVT_Loop(self, structure_filename=None, temp=None, specie="Li"):
        # -- load loop queue script
        from CCpy.Package.Diffusion.SIESTA_NVTLoopQueScript import NVTLoopQueScriptString
        script_string = NVTLoopQueScriptString()
        script_filename = ".AIMDLoop.py"
        f = open(script_filename, "w")
        f.write(script_string)
        f.close()

        jobname = "SNVT%s_%dK" % (structure_filename.replace(".cif", ""), temp)

        mpi = '''#!/bin/csh
# Job name 
#$ -N %s

# pe request
%s

# queue name
%s

# node
%s

#$ -V
#$ -cwd


%s %s %s %s %s
''' % (jobname, self.pe_request, self.queue_name, self.node_assign, self.python_path,
        script_filename, structure_filename, temp, specie)

        f = open("mpi.sh", "w")
        f.write(mpi)
        f.close()

        shl(self.qsub + " mpi.sh", shell=True)
        shl("rm -rf ./mpi.sh", shell=True)
