# Scripts short description

1. Read downloaded data + tips for downloading: [read_smap_smos_3dmean_overlap.py](read_smap_smos_3dmean_overlap.py)
2. Calculate seasonal cycle for the features (TB) and the target (soil moisture) variables: [calculate_seasonal_cycle.py](calculate_seasonal_cycle.py)
3. NN hyperparameters tuning: [sherpa_parameters_search.py](sherpa_parameters_search.py)
4. Regular NN training: [NNsmapsmos_training.py](NNsmapsmos_training.py)
5. Transfer learning (additional NN training): [transfer_learning.py](transfer_learning.py)
6. Combine multiple NN training outputs to calculate structural uncertainty: [structural_uncertainty.py](structural_uncertainty.py)
7. Add noise to input and forward run pretrained NN to get data uncertainty: [data_uncertainty.py](data_uncertainty.py) 
8. Combine all outputs, save full dataset: [combining_outputs.py](combining_outputs.py) 
9. Compare to in-situ SM: [compare_casm_to_insitu.ipynb](compare_casm_to_insitu.ipynb) - **runs with sample data provided**
10. Plot figures: [plot_figures.ipynb](plot_figures.ipynb) - **runs with most of the data provided** 
