def read_data(url):

	f = open(url)
	line = f.readline()
	while line:
		print line
		break
		line = f.readline()
	f.close()

if __name__ == '__main__':
	url = './NSL_KDD'
	read_data(url)