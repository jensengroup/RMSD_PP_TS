%mem=16GB
%nprocshared=4
#irc=(forward calcfc, maxpoint=100, stepsize=5) ub3lyp/6-31G(d,p) scf=(maxcycles=1024)

something title

0 1
N 1.11214 -0.71502 -0.17570
B 1.33386 0.71452 -0.14368
N -1.70156 0.03210 -0.24201
B -0.62410 0.11747 0.67627
H -1.85646 0.69954 -0.98022
H 1.08241 1.34457 -1.12717
H 2.06288 1.15406 0.69306
H 1.54696 -1.34122 0.48564
H 0.74916 -1.18258 -0.99298
H -2.36331 -0.72669 -0.23416
H -0.59580 -0.52769 1.67923
H -0.04869 1.20050 0.73764

