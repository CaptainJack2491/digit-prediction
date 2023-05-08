import os
import base64
import io
import numpy as np
import pandas as pd
import tensorflow as tf
# from tensorflow.keras import layers
from matplotlib import pyplot as plt
# import pygame
from PIL import Image, ImageOps
from flask import Flask, render_template, request

(x_train,y_train),(x_test,y_test) = tf.keras.datasets.mnist.load_data()
x_train_normalized = x_train / 255.0
x_test_normalized = x_test / 255.0

#ploting function
def plot_curves(epochs, hist, list_of_metrics):

    plt.figure()
    plt.xlabel("Epochs")
    plt.ylabel("Value")

    for m in list_of_metrics:
        x = hist[m]
        plt.plot(epochs[1:],x[1:], label=m)
    
    plt.legend()

def create_model(learning_rate):
    """Create and compile a deep neural network"""

    model = tf.keras.Sequential()

    #input layer
    model.add(tf.keras.layers.Flatten(input_shape=(28,28)))

    #hidden layers
    model.add(tf.keras.layers.Dense(units=256, activation="relu"))
    model.add(tf.keras.layers.Dropout(rate=0.4))

    #output layer
    model.add(tf.keras.layers.Dense(units=10, activation='softmax'))

    #compiling model

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

def train_model(model, train_features, train_labels, epochs, batch_size=None, validation_split=0.1):
    """Train the model on the given data"""

    history = model.fit(x=train_features, y=train_labels, batch_size=batch_size,
                        epochs=epochs, shuffle=True, validation_split=validation_split)
    
    epochs = history.epoch
    hist = pd.DataFrame(history.history)

    return epochs, hist

def convert_img(image_data):
    # Extract the base64-encoded data from the data URI
    data = image_data.split(',')[1]

    # Decode the base64-encoded data into a byte string
    decoded_data = base64.b64decode(data)

    # Create a BytesIO object from the decoded byte string
    bytes_io = io.BytesIO(decoded_data)

    #convert and resize the image
    img = Image.open(bytes_io)
    background = Image.new('L',img.size,255)
    background.paste(img,img)
    img = background.resize((28,28))

    arr = np.array(img)
    arr = 255 - arr

    arr = np.expand_dims(arr,axis=0)

    return arr



# def test():
#     # initialize Pygame
#     pygame.init()

#     # set the dimensions of the drawing window
#     WINDOW_SIZE = (280, 280)

#     # create the drawing window
#     screen = pygame.display.set_mode(WINDOW_SIZE)
#     pygame.display.set_caption("Draw a digit")

#     # define the colors
#     BLACK = (0, 0, 0)
#     WHITE = (255, 255, 255)

#     # set the default brush size and color
#     brush_size = 15
#     brush_color = BLACK

#     # create a surface to draw on
#     canvas = pygame.Surface((WINDOW_SIZE[0], WINDOW_SIZE[1]))

#     # set the font for the instructions
#     font = pygame.font.SysFont("Arial", 20)

#     # create a label to display instructions
#     instructions = font.render("Draw a digit and press 'c' to clear or 'Enter' to save", True, BLACK)

#     # set the position of the instructions
#     instructions_rect = instructions.get_rect()
#     instructions_rect.center = (WINDOW_SIZE[0] // 2, 20)

#     # start the Pygame event loop
#     running = True
#     while running:
#         # check for events
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 running = False
#             elif event.type == pygame.MOUSEBUTTONDOWN:
#                 # if the left mouse button is clicked, start drawing
#                 if event.button == 1:
#                     pygame.draw.circle(canvas, brush_color, event.pos, brush_size)
#             elif event.type == pygame.MOUSEMOTION:
#                 # if the mouse is moved while the left button is held down, draw a line
#                 if event.buttons[0]:
#                     pygame.draw.circle(canvas, brush_color, event.pos, brush_size)
#             elif event.type == pygame.KEYDOWN:
#                 # if 'c' is pressed, clear the canvas
#                 if event.key == pygame.K_c:
#                     canvas.fill(WHITE)
#                 # if Enter is pressed, save the image and exit
#                 elif event.key == pygame.K_RETURN:
#                     running = False

#         # draw the instructions and canvas to the screen
#         screen.fill(WHITE)
#         screen.blit(instructions, instructions_rect)
#         screen.blit(canvas, (0, 0))

#         # update the display
#         pygame.display.update()

#     # convert the canvas to a grayscale image
#     gray_img = pygame.surfarray.array2d(canvas)
#     fliped_img = gray_img.transpose()
#     img = Image.fromarray(fliped_img).convert('L')
#     img.save('img.png')
#     resized_img = img.resize((28,28))
#     arr = np.array(resized_img)
#     arr = 255 - arr
#     arr = arr / 255.0
#     arr = np.expand_dims(arr, axis=0)
#     # quit Pygame
#     pygame.quit()

#     return arr

learning_rate = 0.03
epochs = 50
batch_size = 4000
validation_split = 0.2

if os.path.isfile('model.h5'):
    my_model = tf.keras.models.load_model('model.h5')
else:

    my_model = create_model(learning_rate=learning_rate)

    epoch,hist = train_model(my_model,train_features=x_train_normalized, train_labels=y_train,
                         epochs=epochs, batch_size=batch_size, validation_split=validation_split)
    list_of_metrics_to_plot = ['accuracy']
    # plot_curves(epoch,hist,list_of_metrics_to_plot)

    # print("\n Evaluate the model against the test set:")
    # my_model.evaluate(x=x_test_normalized, y=y_test, batch_size=batch_size)

    my_model.save('model.h5')

#flask code
app = Flask(__name__)

@app.route('/', methods=['GET','POST'])
def home():
    if request.method == 'POST':
        # Handle the submitted image data
        image_data = request.form['image_data']
        converted_img = convert_img(image_data)
        print(converted_img)
        # Use the pre-trained model to predict the digit
        digit = my_model.predict(converted_img)
        
        # Convert the predicted digit to a string
        digit_str = str(np.argmax(digit)+1)

        return f"The digit is {digit_str}"
    
    else:
        return render_template('home.html')
        

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)



# prediction = my_model.predict(test())
# for i in prediction:
#     count = 0
#     for j in i:
#         print(count,':',(j*100).round(),'%')
#         count += 1