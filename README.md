# ClathPLM: A Clathrin Prediction Model Combining Multi-Source Protein Language Models with Deep Feature Extraction Strategies

Clathrin is a highly conserved trimeric structural protein capable of self-assembling into a cage-like structure, mediating clathrin-mediated endocytosis and intracellular directional transport. This process is essential for nutrient uptake, signal transduction, synaptic vesicle recycling, and other key functions. Therefore, functional abnormalities in Clathrin are closely associated with the occurrence and progression of various major diseases, such as neurodegenerative diseases, type 2 diabetes, and cancer, highlighting its critical role in basic biological and disease mechanism research. In this study, we propose the ClathPLM model—a robust and highly interpretable deep learning model designed for accurate Clathrin prediction.

![c55b3891fc42604c92f8b7103d4aecfe](https://github.com/user-attachments/assets/f7246f00-456c-44b0-a1f9-0140043a65df)



ClathPLM relies on a large-scale pre-trained protein language models: ProtT5,ProtBert,ESM-3. For detailed guidance on generating protein embedding representations, please refer to the official documentation available at the following websites:

- ProtT5,ProtBert：https://github.com/agemagician/ProtTrans
- ESM-3:https://github.com/evolutionaryscale/esm


## Test on the model

### 1. Prepare Test Data and Labels

Ensure your test data and corresponding labels are ready and match the required input format for the model. You can set them up in the `getdata.py` script.

### 2. Download the Model Weights

Our model can be download from : https://drive.google.com/file/d/1FH5qLeZzTk9XIk6wxh1iMoBiUbQHalj-/view?usp=sharing.

### 3. Run the Test Script
To test the model, run the following command:
```bash
python test.py
```
