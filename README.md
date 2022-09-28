# CASM-dataset

## CASM: A long-term Consistent Artificial-intelligence based Soil Moisture dataset based on machine learning and remote sensing

by Olya Skulovich, Pierre Gentine, Columbia University

Scripts used to create CASM dataset

The dataset is available at https://doi.org/10.5281/zenodo.7072511

The Consistent Artificial Intelligence (AI)-based Soil Moisture (CASM) dataset is a global, consistent, and long-term, remote sensing soil moisture (SM) dataset created using machine learning. It is based on the NASA Soil Moisture Active Passive (SMAP) satellite mission SM data as a target and is aimed at extrapolating SMAP-like quality SM data back in time with previous satellite microwave platforms. Machine learning approach, such as neural network (NN) has the advantage of being both nonlinear, and state-dependent, and naturally imposing a global distribution matching between the source and the target data. Utilizing this, the new CASM dataset was created using high-quality SMAP SM as a target and Soil Moisture and Ocean Salinity (SMOS) or Advanced Microwave Scanning Radiometer - Earth Observing System (AMSR-E/2) brightness temperature as a source, which allowed extrapolating SM data 13 years back from before SMAP mission launch. CASM represents SM in the top soil layer, defined on a global 25 km EASE-2 grid and covers 2002-2020 with a 3-day temporal resolution. The resulting dataset exhibits excellent spatial and temporal homogeneity, without compromising the interannual variability, and is in excellent agreement with the SMAP data (with a mean correlation of 0.97 between the SMAP and CASM SM for the period when the two overlap). Moreover, the input and target datasets were divided into seasonal cycle and residuals, with the NN trained on the residuals. This approach ensures that the high performance does not mask a simple seasonal cycle matching but rather exemplifies the skill targeted at predicting extremes; with the NN achieving a correlation of 0.75 on the test data for the residuals. Comparison to 367 global in-situ SM monitoring sites shows a SMAP-like median correlation of 0.66 between station SM and CASM SM from the corresponding grid cell. Additionally, the SM product uncertainty was assessed, and both aleatoric and epistemic uncertainties were estimated and included in the dataset. Mean epistemic uncertainty, related to the NN model structure, ranges from 0.007 m3/m3 to 0.014 m3/m3 and on average is close to a desired SM product stability threshold of 0.01 m3/m3 per year. Aleatoric uncertainty, defined as input noise propagated through the system, depends on the introduced level of noise. With 10% noise applied to the residuals, the resulting mean standard deviation of the model outputs rises from 0.005 to 0.007 m3/m3.   

The dataset was created with the following steps. Computationally heavy and/or memory intensive steps are highlighed with * and were performed om Columbia University HPC cluster.

1. Download SMAP, SMOS, AMSR-2, E data, ISMN data.

2.* Regrid SMAP data (from command line, using gdalwarp to EASE-2 grid).

3.* Compute seasonal cycle per location for SMAP SM, SMOS and AMSR TB.

4.* Train SMOS->SMAP NN; train AMSR->SMAP SM, perform transfer learning. 

5.* Repeat 4 to get structural uncertainty.

6.* Perform forvard runs with noisy input to get data uncertainty.

7.* Finalize the resulting dataset for analysis and comarison to in situ measurements

8. Analyze the CASM dataset

9. Compare to in-situ SM
