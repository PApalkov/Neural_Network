
from tkinter import *
from Neural_network import NeuralNetwork
from PIL import Image, ImageDraw, ImageFilter


class Paint(Frame):
    def __init__(self, parent):
        Frame.__init__(self, parent)
        self.parent = parent

        self.draw_field_width = 400
        self.draw_field_height = 400
        self.brush_size = 15
        self.color = "black"

        self.image = Image.new("RGB", (self.draw_field_width, self.draw_field_height), "white") #проверить правильность работы
        self.put_on_image = ImageDraw.Draw(self.image)

        self.setUI()

        self.neural_net = NeuralNetwork(inputnodes=28 * 28, hiddennodes=300, outputnodes=10, learningrate=0.01)

    def setUI(self):

        #set sindow size
        self.parent.title("Bumbers Recognizer")

        #put active elements in parent's window
        self.pack(fill = BOTH, expand = 1)

        self.columnconfigure(3, weight=1)
        self.rowconfigure(6, weight=1)

        #making draw field and making white background
        self.canv = Canvas(self, bg="white", width=self.draw_field_width, height=self.draw_field_height)
        self.canv.config(highlightbackground="black")

        self.canv.grid(columnspan=2, rowspan=6,
                           padx=10, pady=10)

        self.canv.bind("<B1-Motion>", self.draw)

        clear_btn = Button(self, text="Очистить", width=30, command=lambda: self.clear())
        save_btn = Button(self, text="Распознать", width=30, command=lambda: self.recognize())

        self.v = StringVar()
        self.v.set("Вы ввели: - ")
        result = Label(self, textvariable = self.v, font=("Helvetica", 25))

        clear_btn.grid(row=0, column=3, padx = 10, pady=10)
        save_btn.grid(row=1, column=3, padx = 10, pady=10)
        result.grid(row = 3, column = 3, padx =10, pady = 10)

    def clear(self):
        self.canv.delete("all")
        self.put_on_image.rectangle((0,0, self.draw_field_width, self.draw_field_height), "white", "white")

    def draw(self, event):
        self.canv.create_oval(event.x - self.brush_size,
                              event.y - self.brush_size,
                              event.x + self.brush_size,
                              event.y + self.brush_size,
                              fill=self.color, outline=self.color)

        self.put_on_image.ellipse((event.x - self.brush_size,
                              event.y - self.brush_size,
                              event.x + self.brush_size,
                              event.y + self.brush_size),'black','black')

    def recognize(self):
        filename = "small_digit.png"
        small_image = self.image.filter(ImageFilter.GaussianBlur(radius=2)).resize((28,28))
        small_image.save(filename)
        answer, probability = self.neural_net.recognize_image(filename)
        if probability < 0.4:
            pass
            self.v.set("Не удалось\nраспознать\nизображение")

        else:
            self.v.set("Вы ввели: " + str(answer))




def main():
    root = Tk()
    root.geometry("650x450+400+400")
    root.resizable(0,0)

    app = Paint(root)
    root.mainloop()


if __name__ == '__main__':
    main()
