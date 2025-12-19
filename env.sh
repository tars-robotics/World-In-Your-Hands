# Install dependencies
pip install h5py matplotlib scipy tqdm open3d
pip install -U "datatrove[ray]"
pip install "laspy[lazrs,laszip]"
pip install pandas
# Install lerobot
cd lerobot
conda install ffmpeg -c conda-forge
pip install -e .
