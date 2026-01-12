# Kidney Vessel Network Analysis
Analysis pipeline for extracting and validating biomarkers from kidney vascular networks.

### Dependencies
python = "^3.12"
numpy = "^2.3.3"
matplotlib = "^3.10.6"
seaborn = "^0.13.2"
scipy = "^1.16.2"
scikit-image = "^0.25.2"
tqdm = "^4.67.1"
skan = "^0.13.0"
pyvista = {extras = ["jupyter"], version = "^0.46.5"}
edt = "^3.1.0"
tifffile = "^2026.1.14"
notebook = "^7.4.7"
jupyterlab = "^4.4.10"
pandas = "^2.3.3"


### Running the Analysis Pipeline
**Plots**
To get the plots in the report run the jupyter notebook plots.ipynb
tree_sim_test.ipynb visualizes the highest strength of each disruption type as in the "bilag"

**Pipeline with rootpainter**
If you download a image dataset from Human Organ Atlas you can convert the .jp2 to .png with img_convert.ipynb so that you can annotate the images in root painter. Afterwards img_convert.ipynb also has function that converts the .png to .tif, which the pipeline uses.

### Hardware
To load in the large dataset you need atleast 32 GB of ram if not more.
A RTX 4070 TI was used to train root painter locally.

