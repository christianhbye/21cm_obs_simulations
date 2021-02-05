import astro as ast
import general as gen
import h5py
import numpy as np
import os

map_file = 'no_git_files/haslam408_ds_Remazeilles2014.fits'
galactic_coord_file = 'no_git_files/pixel_coords_map_ring_galactic_res9.fits'

parent_save = 'no_git_files/sky_models/'
antenna = bd
ground_plane = True
loc = 'mars'
if antenna == bd:
    fs = 'blade_dipole'
    if ground_plane:
        ss = 'inf_metal_ground_plane'
        beam_file = 'no_git_files/blade_diploe_infinite_ground_plane.ra1'
    else:
        ss = 'no_ground_plane'
        beam_file = 'no_git_files/blade_dipole.out'
    path = fs + '/' + ss
save_folder = parent_save + path + '/' + loc 

work_dir = '$SCRATCH/21cm_obs_simulations'

map_orig, lon, lat = ast.map_remazeilles_408MHz(map_file, galactic_coord_file)

if not os.path.exists(save_folder) or not 'save_file_hdf5' in os.listdir(save_folder):
    # Edges = -26.714778, MARS = 79.5
    if loc == 'mars':
        ld = 79.5
    elif loc == 'edges':
        ld = -26.714778
    LST, AZ, EL = ast.galactic_to_local_coordinates_24h_LST(lon, lat, INST_lat_deg= ld, INST_lon_deg=116.605528, seconds_offset=359)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    os.chdir(save_folder)
    with h5py.File('save_file_hdf5', 'w') as hf:
        hf.create_dataset('LST', data = LST)
        hf.create_dataset('AZ',  data = AZ)
        hf.create_dataset('EL',  data = EL)
    os.chdir(work_dir)

start_angle = 0 # first azimuth angle
delta_phi = 30 # when sweeping azimuth angles, phi increments by this number each iteration
N_angles = int((180 - start_angle)/delta_phi)
# master_lst = []
# master_freq = []
# master_conv = []
# master_ant_temp = []

print('Start, delta, N:')
print(start_angle)
print(delta_phi)
print(N_angles)


for i in range(N_angles):
    azimuth = i * delta_phi + start_angle
    print('azimuth = {}'.format(azimuth))
    if beam_file[-4:] == '.out':
        freq_array_X, AZ_beam, EL_beam, Et_shifted, Ep_shifted, gain_shifted = gen.read_beam_FEKO(beam_file, azimuth)
    elif beam_file[-4:] == '.ra1':
        freq_array_X, AZ_beam, EL_beam, Et_shifted, Ep_shifted, gain_shifted = gen.read_beam_WIPLD(beam_file, azimuth)   
    freq_array_X /= 1e6 # convert to MHz

    lst_az_el_file = save_folder+'/save_file_hdf5'
    print(freq_array_X.shape)
    print(AZ_beam.shape)
    print(gain_shifted.shape)
    # Beam
    beam_all_X = np.copy(gain_shifted)
    FLOW         = 40 
    FHIGH        = 120
    freq_array   = freq_array_X[(freq_array_X >= FLOW) & (freq_array_X <= FHIGH)]
    beam_all     = beam_all_X[(freq_array_X >= FLOW) & (freq_array_X <= FHIGH), :, 0:-1] # cut out last col because 0 = 360
    AZ_beam = AZ_beam[0:-1] # cut out last column, same as for beam_all
    print('Sky model')
    # Sky model
    map_freq = 408
    sky_model = ast.map_power_law(map_orig, map_freq, lon, lat, freq_array, spectral_index_lat_model='step', lat_edge_deg=10, index_inband=2.5, index_outband=2.5, sigma_deg=8.5, index_center=2.4, index_pole=2.65)
    print('Local coords')
    # Local Coordinates
    LST, AZ_lst, EL_lst = gen.read_hdf5_LST_AZ_EL(lst_az_el_file)
    print('Convolution')
#   lst_out, freq_out, ant_temp_out = ast.parallel_convolution(LST, freq_array, AZ_beam, EL_beam, beam_all, AZ_lst, EL_lst, sky_model, 40, normalization='yes', normalization_solid_angle_above_horizon_freq=1)
    lst_out, freq_out, conv_out, ant_temp_out = ast.parallel_convolution(LST, freq_array, AZ_beam, EL_beam, beam_all, AZ_lst, EL_lst, sky_model, 40, normalization='yes', normalization_solid_angle_above_horizon_freq=1)
#   master_lst.append(lst_out)
#   master_freq.append(freq_out)
#   master_conv.append(conv_out)
#   master_ant_temp.append(ant_temp_out)

# master_lst = np.array(master_lst)
# master_freq = np.array(master_freq)
# master_conv = np.array(master_conv)
# master_ant_temp = np.array(master_ant_temp)

    with h5py.File(save_folder+'/save_parallel_convolution_MARS_'+str(azimuth), 'w') as hf:
        hf.create_dataset('LST_out', data = lst_out)
        hf.create_dataset('freq_out',  data = freq_out)
        hf.create_dataset('conv_out',  data = conv_out)
        hf.create_dataset('ant_temp_out',  data = ant_temp_out)
