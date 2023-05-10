'''
EN.520.433 Medical Image Analysis
Spring 2023

Step01: Image loading and display
Load all the given data and display the images in subplots

Updated 05.05.2023

Joy Yeh
'''
import SimpleITK as sitk
import matplotlib.pylab as plt

# 0. Basic Control Panel
patient_idx = 20
heartbeat_state = 'ES'  # set to 'ED' or 'ES' to load the corresponding files
folder_name = "data/patient" + str(patient_idx).zfill(4) + "/"
display_markings = True
display_sequence = False
channel_number = 4

# 1. Open the .cfg file
cfg_name = 'Info_2CH.cfg'
with open(folder_name + cfg_name, 'r') as f:
    # Read the lines of the file into a list
    lines = f.readlines()

# Extract the relevant information
age = float(lines[0].split(':')[1].strip())
weight = float(lines[1].split(':')[1].strip())
gender = lines[2].split(':')[1].strip()
edv = lines[3].split(':')[1].strip()
esv = lines[4].split(':')[1].strip()
ef = lines[5].split(':')[1].strip()

# Print the extracted information
print("Information of Patient #" + str(patient_idx).zfill(4))
print("Age:", age)
print("Weight:", weight)
print("Gender:", gender)
print("EDV:", edv)
print("ESV:", esv)
print("EF:", ef)

if display_markings == True:
    # construct the filenames based on the heartbeat_state
    mhp_2ch = 'patient{0:04d}_2CH_{1}.mhd'.format(patient_idx, heartbeat_state)
    mhp_4ch = 'patient{0:04d}_4CH_{1}.mhd'.format(patient_idx, heartbeat_state)
    mhp_2ch_gt = 'patient{0:04d}_2CH_{1}_gt.mhd'.format(patient_idx, heartbeat_state)
    mhp_4ch_gt = 'patient{0:04d}_4CH_{1}_gt.mhd'.format(patient_idx, heartbeat_state)

    # load the images and ground truth files
    I_2ch = sitk.GetArrayFromImage(sitk.ReadImage(folder_name + mhp_2ch, sitk.sitkFloat32))
    I_4ch = sitk.GetArrayFromImage(sitk.ReadImage(folder_name + mhp_4ch, sitk.sitkFloat32))
    I_2ch_gt = sitk.GetArrayFromImage(sitk.ReadImage(folder_name + mhp_2ch_gt, sitk.sitkFloat32))
    I_4ch_gt = sitk.GetArrayFromImage(sitk.ReadImage(folder_name + mhp_4ch_gt, sitk.sitkFloat32))

    # create a figure with subplots arranged in a 2x4 grid
    fig, axs = plt.subplots(2, 4, figsize=(10, 5))
    plt.gray()
    plt.subplots_adjust(0, 0, 1, 1, 0.01, 0.01)

    # display each slice of the images side by side in the same subplot
    for i in range(min(I_2ch.shape[0], 2*4)):
        axs[i//4, i%4].imshow(I_2ch[i])
        axs[i//4, i%4].axis('off')
        axs[i//4, i%4+1].imshow(I_4ch[i])
        axs[i//4, i%4+1].axis('off')
        axs[i//4+1, i%4].imshow(I_2ch_gt[i])
        axs[i//4+1, i%4].axis('off')
        axs[i//4+1, i%4+1].imshow(I_4ch_gt[i])
        axs[i//4+1, i%4+1].axis('off')

    # display the figure
    plt.setp(axs, xticks=[], yticks=[])
    plt.suptitle('Patient {0}, {1} sequence'.format(patient_idx, heartbeat_state))
    plt.show()

if display_sequence == True:
    # construct the filename based on the channel_number
    sequence_file = 'patient{0:04d}_{1}CH_sequence.mhd'.format(patient_idx, channel_number)

    # load the sequence of images
    I_seq = sitk.GetArrayFromImage(sitk.ReadImage(folder_name + sequence_file, sitk.sitkFloat32))

    # display the images using matplotlib
    fig, axs = plt.subplots(1, I_seq.shape[0], figsize=(25, 25))
    for i in range(I_seq.shape[0]):
        axs[i].imshow(I_seq[i], cmap='gray')
        axs[i].axis('off')

    plt.suptitle('Patient {0}, {1}CH Sequence'.format(patient_idx, channel_number))
    plt.show()
