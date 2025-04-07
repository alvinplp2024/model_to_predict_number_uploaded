from flask import Flask, request, jsonify, render_template_string
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import json
from io import BytesIO
import base64

app = Flask(__name__)

# Load the trained CNN model.
try:
    model = load_model("cnn_model.h5")
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Home route: displays a form that allows file upload or drawing.
@app.route("/", methods=["GET"])
def home():
    return render_template_string('''
    <!doctype html>
    <html>
      <head>
        <title>Digit Prediction</title>
        <style>
          body {
            display: flex;
            align-items: center;
            justify-content: center;
            flex-direction: column;
            height: 100vh;
            margin: 0;
            background-color: #f0f0f0;
            font-family: Arial, sans-serif;
          }
          h1, h3 {
            color: #333;
          }
          form {
            background: #fff;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            text-align: center;
            max-width: 500px;
            width: 90%;
          }
          input[type="file"] {
            margin: 10px 0;
          }
          input[type="submit"], button {
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 10px 20px;
            cursor: pointer;
          }
          input[type="submit"]:hover, button:hover {
            background-color: #45a049;
          }
          p {
            color: #666;
            margin-top: 20px;
          }
          #drawCanvas {
            border: 1px solid #000;
            background-color: black;
            cursor: crosshair;
          }
          .selector {
            margin-bottom: 15px;
          }
        </style>
        <script>
          // Toggle between upload and draw methods.
          function toggleMethod() {
            var method = document.querySelector('input[name="method"]:checked').value;
            if (method === "upload") {
              document.getElementById("fileDiv").style.display = "block";
              document.getElementById("canvasDiv").style.display = "none";
            } else {
              document.getElementById("fileDiv").style.display = "none";
              document.getElementById("canvasDiv").style.display = "block";
            }
          }
          
          // Initialize the canvas for drawing.
          function initCanvas() {
            var canvas = document.getElementById("drawCanvas");
            var ctx = canvas.getContext("2d");
            // Fill the canvas with black background.
            ctx.fillStyle = "black";
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            // Set drawing style to white.
            ctx.strokeStyle = "white";
            ctx.lineWidth = 15;
            ctx.lineJoin = "round";
            ctx.lineCap = "round";
            var drawing = false;
            canvas.addEventListener("mousedown", function(e) {
              drawing = true;
              ctx.beginPath();
              ctx.moveTo(e.offsetX, e.offsetY);
            });
            canvas.addEventListener("mousemove", function(e) {
              if (drawing) {
                ctx.lineTo(e.offsetX, e.offsetY);
                ctx.stroke();
              }
            });
            canvas.addEventListener("mouseup", function(e) {
              drawing = false;
            });
            canvas.addEventListener("mouseleave", function(e) {
              drawing = false;
            });
          }
          
          // Clear the canvas and reinitialize it with a black background.
          function clearCanvas() {
            var canvas = document.getElementById("drawCanvas");
            var ctx = canvas.getContext("2d");
            ctx.fillStyle = "black";
            ctx.fillRect(0, 0, canvas.width, canvas.height);
          }
          
          // Convert the canvas drawing to a data URL.
          function submitDrawing() {
            var canvas = document.getElementById("drawCanvas");
            var dataURL = canvas.toDataURL("image/png");
            document.getElementById("canvas_image").value = dataURL;
          }
          
          window.onload = function() {
            toggleMethod();
            initCanvas();
          };
        </script>
      </head>
      <body>
        <h3>ALVIN DATA SCIENCE AND AI</h3>
        <h2>Upload or Draw a Digit</h2>
        <form id="uploadForm" action="/predict_upload" method="post" enctype="multipart/form-data">
          <div class="selector">
            <input type="radio" name="method" value="upload" onclick="toggleMethod()" checked> Upload Image &nbsp;&nbsp;
            <input type="radio" name="method" value="draw" onclick="toggleMethod()"> Draw Digit
          </div>
          <div id="fileDiv">
            <input type="file" name="file" accept="image/*">
          </div>
          <div id="canvasDiv" style="display:none;">
            <canvas id="drawCanvas" width="280" height="280"></canvas>
            <br>
            <button type="button" onclick="clearCanvas()">Clear</button>
                                  
            <!-- Hidden input to store canvas image data -->
            <input type="hidden" name="canvas_image" id="canvas_image">
          </div>
          <br>
          <input type="submit" value="Upload and Predict" onclick="if(document.querySelector('input[name=method]:checked').value==='draw'){ submitDrawing(); }">
        </form>
        <p><b>Note:</b> Uploaded images will be resized to 28x28 pixels.<br>
          In drawing mode, please draw your digit on the black canvas (your strokes will appear in white).
        </p>
      </body>
    </html>
    ''')

# Upload endpoint: accepts an image (either via upload or drawn) and returns the prediction result.
@app.route("/predict_upload", methods=["POST"])
def predict_upload():
    try:
        # Determine whether an image was drawn or uploaded.
        if "canvas_image" in request.form and request.form["canvas_image"]:
            # Process the drawn image (data URL format).
            data_url = request.form["canvas_image"]
            # Data URL format: "data:image/png;base64,...."
            header, encoded = data_url.split(",", 1)
            image_bytes = base64.b64decode(encoded)
            image = Image.open(BytesIO(image_bytes))
            image = image.convert("L")
            image = image.resize((28, 28))
            # For display purposes, we'll reuse the original data URL.
            image_data = data_url
        elif "file" in request.files and request.files["file"].filename != "":
            file = request.files["file"]
            image = Image.open(file.stream)
            image = image.convert("L")
            image = image.resize((28, 28))
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            image_data = "data:image/png;base64," + img_str
        else:
            return jsonify({"error": "No image provided."}), 400

        # Preprocess the image for prediction.
        image_array = np.array(image)
        image_array = image_array.reshape(28, 28, 1).astype("float32") / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        # Run the model prediction.
        prediction = model.predict(image_array)
        predicted_class = int(np.argmax(prediction, axis=1)[0])
        confidence = float(np.max(prediction, axis=1)[0])
        
        # Create a user-friendly message.
        message = f"The image is predicted to be digit {predicted_class} with {round(confidence * 100, 2)}% confidence."
        
        # Build the result JSON (including raw prediction).
        result = {
            "predicted_class": predicted_class,
            "confidence": round(confidence * 100, 2),
            "raw_prediction": prediction.tolist()
        }
        result_json = json.dumps(result, indent=2)
        
        # Create a styled HTML response that shows the image and the prediction.
        html_response = '''
        <!doctype html>
        <html>
          <head>
            <title>Prediction Result</title>
            <style>
              body {
                display: flex;
                align-items: center;
                justify-content: center;
                flex-direction: column;
                height: 100vh;
                margin: 0;
                background-color: #f0f0f0;
                font-family: Arial, sans-serif;
              }
              .result {
                background: #fff;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                text-align: center;
                max-width: 500px;
                margin: 0 auto;
              }
              .result img {
                max-width: 200px;
                border: 1px solid #ddd;
                border-radius: 4px;
                margin-bottom: 15px;
              }
              pre {
                text-align: left;
                background: #f8f8f8;
                padding: 10px;
                border-radius: 4px;
                overflow-x: auto;
              }
              a {
                text-decoration: none;
                color: #4CAF50;
                display: inline-block;
                margin-top: 20px;
              }
            </style>
          </head>
          <body>
            <div class="result">
              <h1>Prediction Result</h1>
              <img src="{{ image_data }}" alt="Uploaded Image">
              <p>{{ message }}</p>
              <pre>{{ result_json }}</pre>
              <a href="/">Upload another image</a>
            </div>
          </body>
        </html>
        '''
        return render_template_string(html_response, message=message, result_json=result_json, image_data=image_data)
    
    except Exception as e:
        error_html = f'''
        <!doctype html>
        <html>
          <head>
            <title>Error</title>
            <style>
              body {{
                display: flex;
                align-items: center;
                justify-content: center;
                height: 100vh;
                margin: 0;
                background-color: #f0f0f0;
                font-family: Arial, sans-serif;
                color: red;
              }}
              .error-container {{
                text-align: center;
              }}
              a {{
                text-decoration: none;
                color: #4CAF50;
                display: inline-block;
                margin-top: 20px;
              }}
            </style>
          </head>
          <body>
            <div class="error-container">
              <h1>Error</h1>
              <p>{str(e)}</p>
              <a href="/">Go Back</a>
            </div>
          </body>
        </html>
        '''
        return error_html, 500

if __name__ == "__main__":
    app.run(debug=True)
