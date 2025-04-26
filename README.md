This repository contains code for the simulations presented in "Coreset selection for the Sinkhorn divergence and generic smooth divergences".
We give many thanks to previously available algorithms from https://github.com/microsoft/goodpoints, https://github.com/satoshi-hayakawa/kernel-quadrature, 
https://github.com/jeanfeydy/geomloss, and https://github.com/alpyurtsever/SKETCH/releases, 
which have been utilized in our experiments, as well as our implementation of CO2-Recombination.

The code in gauss.py represents our typical implementation, where data is generated from an isotropic Gaussian, the entropic optimal transport plan from 
the empirical data to itself is computed, and a Nystrom approximation is made for the resulting kernel. The approximated spectral data is then passed through
a recombination algorithm, resulting in the convexly weighted coreset from our procedure. Other notable files include gauss_grid.py where this simulation is
performed on a Gaussian mixture, cube_samp.py where it is performed on a cube and compared to other sampling algorithms, and mnist_example.py where it is
performed on the MNIST dataset. Alongside these simulations, we include our code for figure generation. 


