from builtins import object, super
from tkinter import *
from tkinter import filedialog as fido
from PIL import ImageTk, Image, ImageDraw
import PIL
from testing import model_test
import io
import numpy as np

classes=[0,1,2,3,4,5,6,7,8,9]
width = 300
height = 300
white = (255, 255, 255)

class App(object):
    DEFAULT_PEN_SIZE = 5.0
    DEFAULT_COLOR = 'black'
    def __init__(self ):
        self.root = Tk()
        self.c = Canvas(self.root, bg='white', width=width, height=height)
        self.c.grid(row=1, columnspan=3)

        self.pil_image = PIL.Image.new("RGB", (width, height), white)
        self.draw = ImageDraw.Draw(self.pil_image)

        self.reset_button = Button(self.root, text='Clear', command=self.reset_drawing)
        self.reset_button.grid(row=0, column=1)

        self.predict_button = Button(self.root, text='Predict Number', command=self.prediction)
        self.predict_button.grid(row = 0, column=2)

        self.setup()
        self.root.title('English Digit Classifier')
        self.root.mainloop()

    def setup(self):
        self.old_x = None
        self.old_y = None
        self.c.bind('<B1-Motion>', self.paint)
        #self.c.bind('<ButtonRelease-1>', self.reset_drawing)

    def reset_drawing(self):
        self.c.delete("all")
        self.old_x = None
        self.old_y = None

    def prediction(self):
        ps = self.c.postscript(colormode='color')
        img = Image.open(io.BytesIO(ps.encode('utf-8')))
        img.save('test.jpg')
        image_name = img
        print(img)
        prediction = model_test(image_name)
        print('argmax',np.argmax(prediction[0]),'\n',
          prediction[0][np.argmax(prediction[0])],'\n',classes[np.argmax(prediction[0])])

    def activate_button(self, some_button):
        self.active_button.config(relief=RAISED)
        some_button.config(relief=SUNKEN)
        self.active_button = some_button

    def paint(self, event):
        self.line_width = 5
        # if self.old_x and self.old_y:
        #     self.c.create_line(self.old_x, self.old_y, event.x, event.y,
        #                       width=self.line_width, fill='black',
        #                       capstyle=ROUND, joinstyle=BEVEL)
        #     self.draw.line(self.old_x, self.old_y, event.x, event.y,
        #                       width=self.line_width, fill='black', smooth=1)
        # self.old_x = event.x
        # self.old_y = event.y
        x1, y1 = (event.x - 10), (event.y - 10)
        x2, y2 = (event.x + 10), (event.y + 10)
        self.c.create_oval(x1, y1, x2, y2, fill="black",width=10)
        self.draw.line([x1, y1, x2, y2],fill="black",width=10,smooth=1)

    def reset(self, event):
        self.old_x, self.old_y = None, None

if __name__ == '__main__':
    App()
