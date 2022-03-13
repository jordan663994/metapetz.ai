# metapetz.ai

# DEPS:

1. tensorflow
2. pandas
3. tensorflow-datasets

# to install(linux), run:

```
sudo apt-get update
sudo apt install python3
sudo apt install python3-pip
pip install tensorflow
pip install pandas
pip install tensorflow-datasets

```
# Windows install instructions:

1. download python here: https://www.python.org/ftp/python/3.7.8/python-3.7.8rc1-amd64.exe and go through the install steps
2. open winInstaller.bat
# Alternate Windows (WSL)

1. open powershell as administrator
2. run this command:
```
wsl --install -d Ubuntu
wsl -d Ubuntu
```
3. run the linux install commands in the wsl console
# to run:

```
python3 Chatbot.py
```
# AI test Documentation

All documentation can be found within documentation.md

----------------------------------------
----------------------------------------

# How your data will be used

when you submit a textpool.textcache file, you are giving me access, and the rights to your conversation with the AI. I will then extract said data (leaving out your username) to compile training data for the AI. After that, the training data will be published to this repository. For details on how this is done, see textpool_processor.py
