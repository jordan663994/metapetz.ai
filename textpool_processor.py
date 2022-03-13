import pandas as pd

f = open("textpool3.textcache", "r")
text = f.read()
f.close()
f = open("textdump.textcache", "r")
text2 = f.read()
f.close()
text_arr = text.split("\n")

text_final = []

for i in text_arr:
	text_final.append(i.split(":"))

text_out = ""
activated = False
name = input("enter user's username")
for i in range(len(text_final)):
	if text_final[i][0] == name:
		text_out += "\n" + "User" + ":" + "\n" + text_final[i + 2][0]
		if activated == False:
			activated = True
	if activated:
		if text_final[i][0] == "Data":
			text_out += "\n" + "Data:" + "\n" + text_final[i + 1][0]
text_out += text2
f = open("textdump.textcache", "w")
f.write(text_out)
f.close()
