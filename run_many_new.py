import os

start_angle = 60
delta = 30
N = (180-start_angle)/delta

for i in range(int(N)):
	phi = int(start_angle + i * delta)
	print('phi = ', phi)
	cmd = 'python run_parallel_conv.py %d' % phi
	os.system(cmd)
