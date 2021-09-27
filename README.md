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
To create the branin datasest:

```console
python branin.py
```

To run the GPR algorithm locally, using the data saved in `branin.arff`. The model will be saved in `gpr.out`.
```console
python gpr.py local
```

Unpickling the result, in a Python console or script.:
```console
pickle.load(open("gpr.pickle", "rb"))
```

Running for OCEAN compute-to-data (assumes "DIDS" is set as an environment variable, input relies in `/data/ddos`, and output is sent to `/data/outputs/result`):
```console
python gpr.py
```

Unpickling is similar to the local version, just change the name of the file to open.


### ARFF
[branin.arff](branin.arff)

### Plot

![Image of branin](branin.png)
