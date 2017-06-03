import numpy as np

protocol_type = {'tcp':0, 'udp':1, 'icmp':2}
flag = {'OTH': 10, 'RSTR': 3, 'S3': 8, 'S2': 9, 'S1': 6, 'S0': 1, 'RSTOS0': 7, 'REJ': 2, 'SH': 4, 'RSTO': 5, 'SF': 0}
service = {'aol': 66, 'urp_i': 19, 'netbios_ssn': 61, 'Z39_50': 14, 'smtp': 15, 'domain': 21, 'private': 2, 'echo': 34, 'printer': 57, 'red_i': 60, 'eco_i': 7, 'ftp_data': 0, 'domain_u': 11, 'urh_i': 64, 'pm_dump': 59, 'uucp_path': 13, 'pop_2': 56, 'systat': 28, 'ftp': 22, 'uucp': 17, 'whois': 31, 'harvest': 69, 'netbios_dgm': 18, 'efs': 30, 'remote_job': 4, 'daytime': 50, 'other': 1, 'finger': 10, 'ldap': 24, 'netbios_ns': 6, 'kshell': 39, 'iso_tsap': 33, 'ecr_i': 25, 'nntp': 46, 'http_2784': 67, 'shell': 51, 'http': 3, 'courier': 47, 'exec': 43, 'tim_i': 58, 'netstat': 52, 'telnet': 9, 'gopher': 26, 'rje': 62, 'sql_net': 40, 'link': 36, 'auth': 20, 'http_443': 29, 'csnet_ns': 16, 'X11': 63, 'IRC': 55, 'tftp_u': 68, 'login': 38, 'pop_3': 53, 'supdup': 12, 'name': 5, 'sunrpc': 37, 'nnsp': 54, 'mtp': 8, 'ntp_u': 44, 'bgp': 23, 'ctf': 48, 'hostnames': 42, 'klogin': 35, 'vmnet': 27, 'time': 41, 'discard': 45, 'imap4': 32, 'http_8001': 65, 'ssh': 49}
label_all = {'guess_passwd': 11, 'spy': 21, 'ftp_write': 12, 'nmap': 6, 'back': 10, 'multihop': 13, 'rootkit': 14, 'pod': 9, 'portsweep': 4, 'perl': 22, 'ipsweep': 3, 'teardrop': 5, 'satan': 7, 'loadmodule': 20, 'buffer_overflow': 15, 'normal': 0, 'phf': 18, 'warezmaster': 17, 'imap': 16, 'warezclient': 2, 'land': 19, 'neptune': 1, 'smurf': 8}
label_part = {'dos': 0, 'u2r': 1, 'r2l': 2, 'probe': 3, 'normal': 5}
label_ca = {'back':'dos', 
'buffer_overflow':'u2r',
'ftp_write':'r2l', 
'guess_passwd':'r2l', 
'imap':'r2l',
'ipsweep':'probe',
'land':'dos',
'loadmodule':'u2r',
'multihop':'r2l',
'neptune':'dos',
'nmap':'probe',
'perl':'u2r',
'phf':'r2l',
'pod':'dos',
'portsweep':'probe',
'rootkit':'u2r',
'satan':'probe',
'smurf':'dos',
'spy':'r2l',
'teardrop':'dos',
'warezclient':'r2l',
'warezmaster':'r2l',
'normal':'normal',
'unknown':'unknown'}

def read_data(url):
	
	feature = np.zeros([22544, 41])
	label_23 = np.zeros([22544, 1])
	label_5 = np.zeros([22544, 1])
	label_2 = np.zeros([22544, 1])
	f = open(url)
	line = f.readline()
	i = 0
	ll = []
	j = 0
	label = {}
	while line:
		
		temp = line.split(',')[:41]
		temp[1] = protocol_type[temp[1]]
		temp[2] = service[temp[2]]
		temp[3] = flag[temp[3]]
		feature[i,:] = temp
		
		l = line.split(',')[41]
		if l == 'normal':
			label_2[i,0] = 0
		else:
			label_2[i,0] = 1
		#label_23[i, 0] = label_all[l]
		#label_5[i, 0] = label_part[label_ca[l]]
		
		i = i + 1
		line = f.readline()
	#print i 
	f.close()
	np.savez('./feature/test.npz', feature=feature, label_2=label_2, label_5=label_5, label_23=label_23)

if __name__ == '__main__':
	url = './NSL_KDD/KDDTrain+.txt'
	read_data(url)


#train size: 125973
#test size: 22544