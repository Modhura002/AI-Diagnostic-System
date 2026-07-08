# pyrefly: ignore [missing-import]
from flask import Flask,render_template,request
import os
from utils.brain_predict import predict_brain
from flask import url_for
from utils.gradcam import generate_gradcam
app=Flask(__name__)
def save_uploaded_file(uploaded_file):

    upload_path = os.path.join(
        "static",
        "uploads",
        uploaded_file.filename
    )

    uploaded_file.save(upload_path) 

    return upload_path

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict",methods=["POST"])
def predict():
    if "image" not in request.files:
        return "No image uploaded"

    uploaded_file = request.files["image"]

    if uploaded_file.filename == "":
        return "Please select an image."
    
    upload_path = save_uploaded_file(uploaded_file)

    result=predict_brain(upload_path)
    generate_gradcam(upload_path)

    return render_template(
        "result.html",

        image_path = url_for(
        "static",
        filename=f"uploads/{uploaded_file.filename}"
        ),
       
        gradcam_path = url_for(
        "static",
        filename=f"results/{uploaded_file.filename}"
        ),

        predicted_class=result["class"],

        confidence=f"{result['confidence']:.2f}"
    )
if __name__=="__main__":
    app.run(debug=True)

