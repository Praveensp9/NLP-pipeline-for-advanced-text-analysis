import time

def load_file(filename):
        file = open(filename,'r')
        text = file.read()
        file.close()
        return text
filename='train.text'
read = load_file(filename)
lines = read.split('\n')
print(type(lines))
for l in lines:
	print l
	time.sleep(10)
