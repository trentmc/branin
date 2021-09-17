# branin
Create branin dataset in ARFF. Plot it too.

### Requirements

- Python 3.8.5+

### Installation
```
#get repo
git clone git clone https://github.com/trentmc/branin.git
cd branin

#install non-virtualenv dependencies
sudo apt-get install python3-tk

#set up virtualenv
python -m venv venv
source venv/bin/activate

#install virtualenv dependencies
pip install wheel
pip install -r requirements.txt
```

### Usage
```console
python branin.py
```

### ARFF
[branin.arff](branin.arff)

### Plot

![Image of branin](branin.png)
