# General Use Guide:

To start, run the install instructions in the readme, then run the start command. Once completed, you will be asked to enter a username (does not matter what you enter). When you have entered your username, you will be able to chat with a group of AI personas you can interact with. After your conversation is over, you will find a file called "textpool3.textcache", please upload that file here: https://discord.com/channels/946095281731166289/951965918689366086.

--------------------------------------


# Conversation Guidelines:

1. When talking to the AI, please say things that are on topic, otherwise, it will ignore you.
2. Please do not spam the AI, it will cause confusion and due to MetaPetz using the uploaded textcache files for future training data this would not help us reach our goals ( since said spam will be containe in that file).
3. No cursing, I shoud not have to say this, but I somehow do.
4. Do not be a bad role model to the AI.
5. Have some fun, explore, see what the AI can do.

---------------------------------------

# Troubleshooting:

1. The best bit of computer troubleshooting advice out there is to restart your PC, this should always be done if there is a strange error that you cannot solve.
2. Are you in the correct directory, if not, use 
```
cd <dirname>
```
3. Do you have a 64 bit python installation? if you have a 32 bit one, you need to uninstall it (which can be done through the python installer) and install the 64 bit one.
4. Are you using a compatable computer, we support 64 bit platforms(the game may support those however because the AI will not be run locally in the game), not ARM, or 32 bit platforms.
5. Still having a problem? Contact me via the metapetz discord here: https://discord.gg/bPb7y9Uc

# If you cannot run winInstaller.bat:

Because a batchfile is literally a bunch of commands, you can run those commands manually in a command prompt, just copy and paste this into your command prompt to do the exact same thing the batch file would do:
```
pip install tensorflow
pip install pandas
pip install tensorflow-datasets
pip install numpy
pip install Pillow
```
