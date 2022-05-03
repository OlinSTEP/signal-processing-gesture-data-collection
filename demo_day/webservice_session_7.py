from flask import Flask, request
import joblib
import numpy as np

app = Flask(__name__)
# model = joblib.load("pca_svm_full_model_session_7.pkl") # low real accuracy, trained on 192 gestures
model = joblib.load("pca_svm_full_model_session_4_and_5_and_7.pkl") # high real accuracy, trained on 1,212 gestures

def save_trace(t_data):
    np.savetxt("example.csv", t_data)

@app.route("/", methods=["POST"])
def predict_gesture():
    if "imuData" not in request.json:
        return "gesture data not found", 400
    imu = np.array(request.json['imuData']) # order: la_x, la_y, la_z, a_1, a_2, a_3, a_4
    imu = np.transpose(imu)

    print(imu.shape)

    X_test = imu.reshape((-1, 700))
    save_trace(X_test)

    pred_test = model.predict(X_test[:, 0:300])

    pred_idx = 0
    if (pred_test[0] == 4): # right (right)
        pred_idx = 0
        print("right: swipe", pred_idx)
    if (pred_test[0] == 2): # down (down)
        pred_idx = 1
        print("down: tap", pred_idx)
    if (pred_test[0] == 3): # counter clockwise from bottom (left)
        pred_idx = 2
        print("back: counter clockwise from bottom", pred_idx)
    if (pred_test[0] == 1): # flick (up)
        pred_idx = 3
        print("up: flick", pred_idx)

    print("pred_idx", pred_idx)

    return {'msg': 'success', "pred_idx": pred_idx}

def main():
    app.run(host='0.0.0.0', debug=True)

if __name__ == "__main__":
    main()
