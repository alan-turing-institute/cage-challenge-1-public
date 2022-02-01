import cv2
import glob
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

from results_parser import parse_results

##########################
# Constants
##########################
# # Image shades
known_shade = (1,0,0,0.5) # facecolor
unknown_shade = (0,0,1,0.5) # facecolor
access_none = 'b' # edgecolor
access_user = 'y' # edgecolor
access_priv = 'r' # edgecolor
server_shape_w = 65
server_shape_h = 85
host_shape_w = 62
host_shape_h = 62

def get_loc(hostname):
    """
    Get the location of the host on the image.

    param:
        hostname: string from API to specify host
    return:
        loc: tuple with position for graphic on specified host
    """
    if hostname == "Enterprise0":
        loc = (460, 55)
    elif hostname == "Enterprise1":
        loc = (532, 55)
    elif hostname == "Enterprise2":
        loc = (603, 55)
    elif hostname == "Op_Server0":
        loc = (968, 55)
    elif hostname == "Op_Host0":
        loc = (908, 315)
    elif hostname == "Op_Host1":
        loc = (973, 315)
    elif hostname == "Op_Host2":
        loc = (1035, 315)
    elif hostname == "User0":
        loc = (0, 317)
    elif hostname == "User1":
        loc = (67, 317)
    elif hostname == "User2":
        loc = (135, 317)
    elif hostname == "User3":
        loc = (204, 317)
    elif hostname == "User4":
        loc = (273, 317)
    elif hostname == "Defender":
        loc = (535, 315) 
    return loc

def add_patch(hostname, known_state, access_state, scanned=False):
    """
    Add the necessary shape to the image

    param:
        hostname: string from API to specify host
        known_state: Boolean indicating if red agent knows the IP of the corresponding host
        access_state: ["None", "User", "Priveleged"] indicating red agent access to corresponding host
        scanned: Boolean indicating if blue agent previously scanned the corresponding host
    return:
        None, shape is added to the image.
    """
    # Translate variables to colors
    if known_state:
        facecolor_shade = known_shade
    else:
        facecolor_shade = unknown_shade
    if access_state == "None":
        edgecolor_shade = access_none
    elif access_state == "User":
        edgecolor_shade = access_user
    else: # access_state == "Privileged"
        edgecolor_shade = access_priv
        
    if "Enterprise" in hostname or "Server" in hostname:
        shape_w = server_shape_w
        shape_h = server_shape_h
    else: # Host
        shape_w = host_shape_w
        shape_h = host_shape_h
    loc = get_loc(hostname)
    
    # Create a Rectangle patch
    rect = patches.Rectangle(loc, shape_w, shape_h, linewidth=3, 
                             edgecolor=edgecolor_shade, facecolor=facecolor_shade)
    # Add the patch to the Axes
    ax.add_patch(rect)


if __name__ == "__main__":
    # Grab logs
    file_name = "logs_to_vis/results.txt" # CHANGE HERE
    results_json = parse_results(file_name)

    # Make images for each step
    for i in range(len(results_json)):
        img = results_json[i]
        base = Image.open('./img/figure1.png')
        # Create figure and axes
        fig, ax = plt.subplots(figsize=(20, 8))
        # Display the image
        ax.imshow(base)

        for host in img["hosts"]:
            # Add the shape to cover host with specified information
            add_patch(host["hostname"], host["known"], host["access"], host["scanned"])
        if i == 0:
            plt.title("Starting...")
        else:
            plt.title(f"Blue Action: {img['blue_action']} \nReward: {img['reward']} \nEp Reward: {img['ep_reward']}")
        plt.axis('off')
        plt.savefig(f"./img/img{i}.png")
        # plt.show()

    # Compile images into a gif
    # filepaths
    fp_in = "img/img*.png"
    fp_out = "img/results.gif" # "./img/" + file_name.lstrip("logs_to_vis/").rstrip(".txt") + ".gif"

    # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
    img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
    img.save(fp=fp_out, format='GIF', append_images=imgs,
             save_all=True, duration=400, loop=0)

