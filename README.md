# InundationTools
Remote sensing-based inundation tools
 
Things to install:

External Programs:
1.	Sentinel 1 Toolbox. http://step.esa.int/main/download/snap-download/
Need the gpt.exe which is downloaded with s1tbx. The path then needs to be added to the code Preprocess_Sentinel_GRDH_data.py to work.
2.	Pyspark will be used to optimize the later codes. It is a python package but needs to be configured external to python.
3.	ISCE is used to process coherence images and will need to be used for that processing. The version of ISCE is 2.3 or the latest version should be fine.

Python Packages:
1.	Most important is Rasterio version 1.1.4 with GDAL version 3.0.4.
2.	I have uploaded a file named environment_droplet.yml which includes all the python packages I have installed under the environment I am using for this project. I would just try and run the code and if a package is missing check what version I have then install it. This environment is a conda environment but also has used pip to install packages.

Currently there are two python files uploaded the first is called Preprocess_Sentinel_GRDH_data.py which is the code that used s1tbx. There is a variable near the top of this code called baseSNAP which needs to be updated to include the path to were gpt.exe is located, or if its already in your search path then just the gpt.exe itself. This code also used gdal_translate at lines 117 and 118 which has a path to this code, so you can either include the direct path or if it is in your search path it just needs to find it as it is sending this command to the terminal. Currently these are the only changes needed in this set of code to get it to work.

Running this set of codes:
The directory structure as it is in github needs to be preserved. In the main folder you need to make a folder called Data, then the zipped data which I will provide two zipped data folders, needs to go into this folder. Once we have some type of input where I can automatically download data this step will be automated, but now it is manual because I don’t have an input to automatically download data yet. The main code to run these codes is a Jupyter notebook called run_all.ipynb. This will run each step that is currently set up. Cell 3 is the first step, you can change pixsiz to 100 here and that should significantly speed up the processing as well as reduce storage space requirements. Nothing else needs to be changed. If the python environment is set up correctly, and all external programs are downloaded and called you should just be able to go through cells 3-7 with no errors. This processing will produce a folder in main called GRD_Processed, were all of the processed data is going and intermediate products being saved. It is currently saving all pre processed products. If everything worked, within the processed folders in GRD_Processed you should see something like this.
