# create env uisng venv
python -m venv neuro_env

# activate env
source ./neuro_env/bin/activate

# upgrade pip
python -m pip install --upgrade pip

# install requirements
python -m pip install -r requirements.txt