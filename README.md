# RMSD_PP_TS
Locate TS based on RMSD-PP method

## Depency on external programs
The procedure relies on external programs to do the quantum chemical calculations. 
In particular, there are calls to four different submit scripts (submitting xTB and Gaussian16 calculations):


### submit_xtb_path:
submits an RMSD-PP GFN2-xTB calculation for
* reactant structure: reactant.xyz
* product structure: product.xyz
* push value: k_push
* pull value: k_pull
* alpha value (width of Gaussian biasing potential): alp
usage:
```
submit_xtb_path reactant.xyz product.xyz k_push k_pull alp
```
### submit_batches_xtb:

### submit_batches_gaussian:

### submit_gaus16:
