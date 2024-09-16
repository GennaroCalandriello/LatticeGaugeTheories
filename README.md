# LatticeGaugeTheories
This repository simulates the behavior of pure SU2 and SU3 lattice gauge theories using Metropolis, Heat Bath and Over Relaxation updating algorithms. Here is one result for N=5 space-time extension of the lattice, for beta varying in range 0.1-8.0 and for 2 different Wilson Loop extensions, referred to a SU(3) theory. You can compare these results with Gattringer

![w11_w22_heatbath](https://user-images.githubusercontent.com/91687268/213214805-c0a8a807-9be4-4e49-a11b-c40491c31a96.png)

For the SU(2) pure gauge the results obtained with the 3 algorithms are the following (Wilson loop calculated for 3 different spacetime extensions: W11, W22, W33):
Heat-Bath + 2 Over-relaxation for each measure
![hbor](https://user-images.githubusercontent.com/91687268/213728339-8e573378-b6e0-405c-99df-e91788706fe0.png)

Metropolis (N_hits=10)
![su2metrohits50](https://user-images.githubusercontent.com/91687268/213728370-2a07dc4c-5bf7-4084-b966-e8d313e43138.png)

This is the first result of Higgs coupling with SU(2), with a comparison (N=7, only heath bath, $\beta$ = 2.2):
![su2 secondo risultato N = 7](https://user-images.githubusercontent.com/91687268/213894384-0305dd98-3623-45bc-b08b-945a5b9118c3.png)
![su2gaugehiggsvaryingk](https://user-images.githubusercontent.com/91687268/213894392-91c8e736-8851-497c-a0a0-bf11cd8bfca2.png)
Here another result for SU(2)- Higgs coupling obtained varying k (Higgs coupling)
![beta_2 5_200_points](https://user-images.githubusercontent.com/91687268/215287564-1f745b9a-df9b-4101-9f41-907532518a37.png)
# Deconfinement transition: Polyakov loops expectation values
![Polyakov loops on the Re-Im plane, signaling the Z3 symmetry breaking with the
generation of 3 distinct vacua. We compute the average of the bare Polyakov loop
on a 74 lattice for 10000 configurations using 1 Heat-Bath and 2 Over-Relaxation updates.](https://github.com/GennaroCalandriello/LatticeGaugeTheories/blob/main/PythonVersion/images/polyakov5.7hbor.png)

