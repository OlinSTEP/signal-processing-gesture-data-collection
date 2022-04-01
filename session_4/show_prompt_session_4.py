import random
import numpy as np
import matplotlib.pyplot as plt
import random
from PIL import Image
import time
import itertools

"""
Show prompt for gesture data collection
- runs for 120 seconds, shows prompt every 3 seconds
- 4 chosen gestures (prompt images stored in `prompts/`)
"""

def show_prompt(key="a", wait=6):
    image_address = f"prompts/{key}.jpg"
    image_itself = Image.open(image_address)
    image_numpy_format = np.asarray(image_itself)
    plt.imshow(image_numpy_format)
    plt.draw()
    plt.pause(wait) # pause how many seconds
    plt.close()

def main():
    # 4 gestures
    gestures = ["b", "d", "f", "i"]
    # 10 repetitions
    reps = 10
    rep_gestures = [list(itertools.repeat(g, reps)) for g in gestures]
    rep_gestures = list(itertools.chain.from_iterable(rep_gestures))

    # randomize order
    random.shuffle(rep_gestures)
    
    # 6 seconds to get ready
    print("Get in position...")
    print("Tap to mark start and end of each gesture...")
    time.sleep(6)

    # show each prompt for 6 seconds
    for g in rep_gestures:
        show_prompt(g, wait=3)
    
    print("-".join(rep_gestures))
    print("Remember to upload data to firebase!")

if __name__ == "__main__":
    main()