# AiCHE2023
Code for "A Graph Attention Network Based Approach for Interpretable and Domain-Aware Modeling of a Wellhead Water Treatment System" (AiCHE 2023)

### Abstract:
Rural communities in agricultural regions across the United States are increasingly confronted with increasing salinity and nitrate levels of their local potable groundwater sources. The water sources of many communities exceed the maximum nitrate contaminant level, and their water salinity is well above the recommended level for drinking. In order to provide safe potable water to these communities, membrane-based wellhead water treatment system was developed and deployed in multiple communities. However, given the remote locations, such wellhead water treatment systems must be autonomous and adapt to handle intermittent system operations due to the fluctuating water use patterns and unavailability of continuous manual labor. Since the operation of such water treatment systems is intermittent (given the fluctuating hourly and daily water usage, and the finite local water storage volume), it is critical to develop a system performance model that is suitable for model-based control, performance forecasting, fault detection, and determination of causal relationships among process attributes. To accomplish the above, an advanced ensemble machine learning approach, based on graph neural networks with attention mechanism (GATs), was developed to describe the intermittent operational profiles of a wellhead water treatment system deployed in a small remote disadvantaged community in California.

### Implementation Details:
This method is based on the idea of Graph Attention Networks (GATs) and borrows heavily in terms of implementation from [A. Deng. et al.](https://github.com/d-ailin/GDN). We implement two versions of the model: one that uses expert knowledge to build an apriori graph structure that the model learns attention weights of, and the other that completely learns a graph structure for the system. Through evaluation on a test dataset, and RMSE values for the prediction of Salt Passage and Permeate Nitrate Concentration for the system, we show that specifying an expert-defined graph structure is useful towards faster convergence to a better predictive model for the system parameters. 

### Raw Data:
The raw data used in this work can be downloaded from this [Google Drive Link](https://drive.google.com/file/d/1c9zPA3zY_nAEzemq2m4tjUBMUoUfyepS/view?usp=sharing) 
Preprocessing:
- The raw data is resampled to one per 5s frequency, and missing values are interpolated. Invalid values (such as Salt Passage % less than 0 or greater than 100) are appropriately handled.
- The raw data by itself does not have shutdown or startup states defined, so additional RO State Shutdown and RO State Startup are defined using the last 1000 RO State 2 values and the first 1000 RO State 2 values.
- The raw data is filtered to include only parameters: 'PT4-HP Concentrate (psig)', 'FTF-Raw Feed Inflow (gpm)',
             'PT2-RO Pump Inlet(psig)', 'PT3-RO Pump Outlet (psig)', 'PT5-Permeate (psig)', 'Permeate Flux (gfd)', 'FT1-Inlet (gpm)',
             'FT2-Recycle (gpm)', 'CT1-Feed (uS)', 'CT2-Permeate (uS)', 'NT1-Permeate (ppm NO3-N)',
             'Salt Passage (%)', 'RO State'. PT3 and PT4 are further replaced by transmembrane pressure (mean(PT3, PT4)).
- Training, Validation, Test Splits are created per RO State and the data is scaled using MinMaxScaler from sklearn.

### Preprocessed Data:
The preprocessed data can be downloaded from this [link](https://drive.google.com/drive/folders/1o3ViDfOzDEfdQwf3FW3bs5Iv0egsDUOx)
Ideally, the data should be in the same directory as the run.sh file with the following format: data/<rostate>_modified/train; data/<rostate>_modified/val; data/<rostate>_modified/test

### Training:
For training the model, follow the following steps:
- Modify the hyperparameters as needed in the run.sh script. Hyperparameters to tune: hidden_dim, out_dim, num_layers, decay 
- To run the training for a certain mode run:
	```sh run.sh cpu 2_modified apriori```
The apriori argument implies that we use the apriori specified graph structure, 2_modified implies that we are training the model for RO state 2(production) with the mean(PT3, PT4) parameter as a feature as opposed to PT3, PT4 separately 
- For other modes run:
```sh run.sh cpu 4_modified apriori```
```sh run.sh cpu 5_modified apriori```
```sh run.sh cpu -1_modified apriori```
```sh run.sh cpu -2_modified apriori```
For the comparison with the completely learnable graph structure, we would run:
```sh run.sh cpu 2_modified learnable``` , etc. 
Ideally we would want to run the training for learnable graph structure after the model with the apriori graph structure is hyperparameter-tuned so we can compare the RMSE for the same hyperparameters for a fair comparison  
