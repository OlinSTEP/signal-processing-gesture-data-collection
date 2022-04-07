import requests
import numpy as np
import json

def main():    
    # load test labels
    f = open('meta_gestures_test.csv', 'r')
    meta = f.readlines()

    # load test data
    data = np.genfromtxt("data_gestures_test.csv")
    
    # get expected test label
    label = meta[0]

    # turn first test into json
    d = data[0] # numpy array
   
    # j_obj = json.dumps(obj)
    # print("Sending:", j_obj)

    url = 'http://0.0.0.0:5000/'
    r = requests.post(url,json={"imuData": d})

    print(r.json())
    print("Expecting:", label)

if __name__ == "__main__":
    main()