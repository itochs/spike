pip install --upgrade pip --user
pip --no-cache-dir install -I tensorflow==1.14.0
pip --no-cache-dir install -I pillow==6.2.1
pip --no-cache-dir install -I pyserial==3.4
pip --no-cache-dir install -I gym==0.16.0
pip --no-cache-dir install -I opencv-python==4.0.0.21
pip --no-cache-dir install -I keras-rl==0.4.2
pip --no-cache-dir install -I matplotlib==3.1.3
pip --no-cache-dir install -I jupyterlab
pip uninstall keras
pip --no-cache-dir install -I keras==2.3.1
pip uninstall h5py
pip uninstall pyyaml
pip --no-cache-dir install -I h5py==2.10.0
pip --no-cache-dir install -I PyYAML==5.1