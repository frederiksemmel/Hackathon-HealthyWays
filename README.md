# HealthyWays

HealthyWays is currently a Python application that lets you find the best public transport route in a given timeframe in order to avoid crowds. It was initiated during the VersusVirus Hackaton in April 2020.

A Team Project of: Agnese Sacchi, Yan Walesch, Frederik Semmel, Jona Bühler and Marino Müller

## Installation

Use the package manager [poetry](https://python-poetry.org/) to set up the evironment for HeahthyWays. Install it as described in the documentation.

Install dependencies with poetry
```bash
poetry install
```

If on Mac you have to use python 3.7 because of tensorflow 2.2.0rc2. Install  [pyenv](https://github.com/pyenv/pyenv) and run the following commands inside the virtual environment

### Initialize pyenv on fish (for bash read the pyenv readme)
```
set -Ux PYENV_ROOT $HOME/.pyenv
set -U fish_user_paths $PYENV_ROOT/bin $fish_user_paths
```
Add the following to the fish config at .config/fish/config.fish
```
# pyenv init
if command -v pyenv 1>/dev/null 2>&1
    pyenv init - | source
end
```
Now restart all the shells and then this should set the python version for a specific directory and all subdirectories
```
pyenv install 3.7.*
pyenv local 3.7.*  # Activate Python 3.7.* for the current project
```
If poetry still shows you the wrong python version, try reinstalling it.


### Google API Key
You also need your own Google API key in order to run the application. You can get it on the Google Cloud Platform

For bash
```bash
export GOOGLE_API_KEY='your api key'
```

For fish
```fish
set -Ux GOOGLE_API_KEY='your api key'
```

## Usage

You should activate the virtual environment like so
```
poetry shell
```

If you are in the virtual environment you can use
```bash
python healthyways
```

If you have not activated the virtual envorionment use
```bash
poetry run python healthyways
```

## Contributing
Use a code formatter. For python we use black.


## License
[MIT](https://choosealicense.com/licenses/mit/)
