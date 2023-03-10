import matplotlib.pyplot as plt
import requests
from Image_conv import load_and_prep_image

def pred_and_plot(model, filename, class_names):
  request = request.get("https://raw.githubusercontent.com/AtulBinu2002/Sort-Your-Plate-CCN-/main/Function/Image_conv.py")
  with open("Image_conv.py", "wb") as f:
    f.write(request.content)
  """
  Imports an image located at filename, makes a prediction on it with
  a trained model and plots the image with the predicted class as the title.
  """

  # Import the target image and preprocess it
  img = load_and_prep_image(filename)

  # Make a prediction
  pred = model.predict(tf.expand_dims(img, axis=0))

  # Get the predicted class
  pred_class = class_names[int(tf.round(pred)[0][0])]

  # Plot the image and predicted class
  plt.imshow(img)
  plt.title(f"Prediction: {pred_class}")
  plt.axis(False)
  plt.show()
