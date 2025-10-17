# importing whole module
from tkinter import *
from tkinter.ttk import *

# importing strftime function to
# retrieve system's time
from time import strftime
from datetime import datetime

# creating tkinter window
root = Tk()
root.title('Clock')

# This function is used to
# display time on the label

def time():
	# string = strftime('%H:%M:%S')
	string = datetime.now().strftime('%H:%M:%S.%f')[:-3]
	lbl.config(text=string)
	lbl.after(10, time)


# Styling the label widget so that clock
# will look more attractive
lbl = Label(root, font=('calibri', 250, 'bold'),
			background='black',
			foreground='white')

# Placing clock at the centre
# of the tkinter window
lbl.pack(anchor='center')
time()

mainloop()
