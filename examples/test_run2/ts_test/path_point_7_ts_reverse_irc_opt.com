%mem=16GB
%nprocshared=4
# opt freq ub3lyp/6-31G(d,p) scf=(maxcycles=1024)

something title

0 1
N -1.67101 -0.55418 -0.04491
B -0.91130 0.66246 -0.02163
N 1.67101 -0.55418 0.04491
B 0.91130 0.66246 0.02163
H 2.63196 -0.55446 -0.25370
H -1.48388 1.70514 0.07029
H -0.03408 0.79068 -0.97597
H -1.27107 -1.47803 -0.08441
H -2.63195 -0.55446 0.25371
H 1.27107 -1.47803 0.08441
H 0.03408 0.79068 0.97596
H 1.48388 1.70514 -0.07029

