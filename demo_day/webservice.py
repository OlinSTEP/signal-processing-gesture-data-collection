from flask import Flask, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("pca_svm_train_model.pkl")

@app.route("/", methods=["POST"])
def predict_gesture():
    if "data" not in request.json:
        return "gesture data not found", 400
    imu = np.array(request.json['imuData']) # order: la_x, la_y, la_z, a_1, a_2, a_3, a_4
    X_test = imu.reshape((-1, 700))

    pred_test = model.predict(X_test[:, 0:300])
    pred_idx = int(pred_test[0])
    
    return {'msg': 'success', "pred_idx": pred_idx}

def main():
    app.run(host='0.0.0.0', debug=True)

if __name__ == "__main__":
    main()