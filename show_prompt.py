import random
import numpy as np
import matplotlib.pyplot as plt
import random
from PIL import Image
import time

"""
Show prompt for gesture data collection
- runs for 90 seconds, shows prompt every 6 seconds
- 14 gestures (prompt images stored in `prompts/`)
"""

def show_prompt(key="a"):
    image_address = f"prompts/{key}.jpg"
    image_itself = Image.open(image_address)
    image_numpy_format = np.asarray(image_itself)
    plt.imshow(image_numpy_format)
    plt.draw()
    plt.pause(6) # pause how many seconds
    plt.close()

def main():
    # 14 gestures
    gestures = ["a", "b", "c", "d", "e",
                "f", "g", "h", "i", "j",
                "k", "l", "m", "n"]
    # randomize order
    random.shuffle(gestures)
    
    # 6 seconds to get ready
    print("Get in position...")
    print("Tap when ready...")
    time.sleep(6)

    # show each prompt for 6 seconds
    for g in gestures:
        show_prompt(g)
    
    print("-".join(gestures))
    print("Tap when finished")
    print("Send data to firebase")

if __name__ == "__main__":
    main()