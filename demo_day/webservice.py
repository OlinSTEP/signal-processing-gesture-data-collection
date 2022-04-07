from flask import Flask, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("pca_svm_train_model.pkl")

@app.route("/", methods=["POST"])
def predict_gesture():
    if "imuData" not in request.json:
        return "gesture data not found", 400
    imu = np.array(request.json['imuData']) # shape (100, 7): la_x, la_y, la_z, a_1, a_2, a_3, a_4
    imu = np.transpose(imu) # shape (7, 100)
    X_test = imu.reshape((-1, 700))

    pred_test = model.predict(X_test[:, 0:300])
    pred_idx = int(pred_test[0])

    pred_idx = 0
    if (pred_test[0] == 2): # right
        pred_idx = 0
    if (pred_test[0] == 4): # right
        pred_idx = 1
    if (pred_test[0] == 6): # right
        pred_idx = 2
    if (pred_test[0] == 9): # right
        pred_idx = 3
    
    return {'msg': 'success', "pred_idx": pred_idx}

def main():
    app.run(host="0.0.0.0", debug=True)

if __name__ == "__main__":
    main()