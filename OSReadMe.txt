INSTALLATION GUIDE FOR ENVISIONHGDETECTOR USING MEDIAPIPE

Use Anaconda (NOT Miniconda)

1. conda create -n ENVNAME python=3.9
2. conda activate ENVNAME
3. conda install pip
4. cd LOCATIONOFREQUIREMENTSTXT
5. pip install -r requirementsmacos.txt
6. pip install envisionhgdetector --no-deps
7. conda install -c conda-forge lightgbm
8. conda install --force-reinstall numpy==1.26.4 pandas
