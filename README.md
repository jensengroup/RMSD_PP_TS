# RMSD_PP_TS
Locate TS based on RMSD-PP method

## Depency on external programs
The procedure relies on external programs to do the quantum chemical calculations. 
In particular, there are calls to four different submit scripts (submitting xTB and Gaussian16 calculations) within rmsd_pp_ts.py:


### submit_xtb_path:
Submits an RMSD-PP GFN2-xTB calculation for
* reactant structure: reactant.xyz
* product structure: product.xyz
* push value: k_push
* pull value: k_pull
* alpha value (width of Gaussian biasing potential): alp  

The file ```path.inp``` must be present in the directory  
When called separately, the usage is as follows:  
```
submit_xtb_path reactant.xyz product.xyz k_push k_pull alp
```
### submit_batches_xtb:
Submits all .xyz files in the current directory to single point energy calculations using the xTB program
The jobs are submitted to slurm in batches  
When called separately, the usage is as follows:  
Go to the directory with the .xyz files to be submitted and call  
```
submit_batches_xtb
```

### submit_batches_gaussian:
Works as ```submit_batches_xtb``` except that the single point energy calculations are done using Gaussian 16, and instead of .xyz files it submits all gaussian input files (.com file) in the present directory. 

### submit_gaus16:
Submits a Gaussian16 calculation with the type of calculation, structure and method stated in the input .com file  
When used separately, the usage is as follows:  
```
submit_gaus16 input.com
```  

The four submit scripts used can be found in submit_scripts
