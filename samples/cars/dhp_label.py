from tkinter import *
import tkinter.messagebox
import csv
import cv2 as cv
import matplotlib.pyplot as plt
from PIL import Image, ImageTk

CSV_DIR='book1.csv'

class Rect:
    idx = -1
    x = None
    y = None
    w = None
    h = None

    def str2float(self, s):
        def char2num(s):
            return {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9}[s]
            # 这事实上是一个字典

        index_point = s.find('.')
        if index_point == -1:
            daichu = 1
        else:
            daichu = 0.1 ** (len(s) - 1 - index_point)
            s = s[0:index_point] + s[index_point + 1:]  # 这里是除去小数点
        from functools import reduce
        result1 = reduce(lambda x, y: x * 10 + y, map(char2num, s))
        return result1 * daichu

    def __init__(self, idx, str):
        self.idx = idx
        l = str.split('_')
        # print(l)
        self.x = int(self.str2float(l[0]))
        self.y = int(self.str2float(l[1]))
        self.w = int(self.str2float(l[2]))
        self.h = int(self.str2float(l[3]))

    def draw_me(self, image):
        cv.rectangle(image, (self.x, self.y), (self.x + self.w, self.y + self.h), (0, 250, 0))

        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(image, str(self.idx), (self.x + int(self.w / 2), self.y + int(self.h / 2)), font, 1, (0, 250, 0), 1,
                   cv.LINE_AA)

    def print_me(self):
        print("x,y, w, h: %d, %d, %d, %d" % (self.x, self.y, self.w, self.h))

class App:
	def __init__(self):

		master = Tk()
		master.title('label me')


		self.label =''
		self.img_list=[]
		self.label_list=[]
		self.cur_idx = 0
		self.img_on_canvas = None

		self.frame = Frame(master)
		self.canvas = Canvas(self.frame, width=1100, height = 520)
		self.canvas.pack(side=LEFT)


		self.frame2 = Frame(master)
		self.frame2.pack()
		self.btn_next = Button(
			self.frame2, text="下一个",width=15, height=3, fg="red", command=self.next_pic
		)

		self.btn_prev = Button(
			self.frame2, text="上一个", width=15, height=3, fg="red", command=self.prev_pic
		)

		frame2 = Frame(master)
		frame2.pack()
		self.label_idx = Label(frame2, text="No.")

		self.label_now = StringVar(value='3:1,4:2')
		self.ent_label=Entry(frame2,textvariable=self.label_now,font=('Verdana',18))
		self.btn_submit = Button(
			frame2, text="保存 ", width=15,height=3,fg="red", command=self.submit_label
		)

		self.btn_save = Button(
			frame2, text="导出", width=15,height=3,fg="red", command=self.write_csv
		)

		self.btn_prev.grid(row=1, column=1)
		self.btn_next.grid(row=1, column=2)
		self.label_idx.grid(row=2, column=1)
		self.ent_label.grid(row=2,column=2)
		self.btn_submit.grid(row=3,column=2)
		self.btn_save.grid(row=3,column= 1)
		
		#read csv, load first image
		self.load_csv(CSV_DIR)
		self.load_image(0)

		master.mainloop()

	def put_image(self, master, idx):
		if idx >= len(self.img_list):
			return
		name = "images/"+self.img_list[idx][0]

		img = cv.imread(name)
		b, g, r = cv.split(img)
		img = cv.merge((r, g, b))

		im = Image.fromarray(img)
		tk_im = ImageTk.PhotoImage(image = im)

		self.label_pic = Label(master, image=tk_im)
		self.label_pic.pack()
		#canvas.create_image(1100, 500, tk_im)

	def load_csv(self, path):
		with open(path) as csv_input:
			fields = ['name', 'bbox']
			reader = csv.DictReader(csv_input, fieldnames=fields)
			for row in reader:
				tp_img = (row['name'], row['bbox'])
				self.img_list.append(tp_img)
	def write_csv(self,):
		
		tkinter.messagebox.showinfo('导出将重写out_label.csv')
		path ="out_label.csv"	
		kwargs = { 'newline':''}
		mode = 'w'
		with open(path, mode, **kwargs) as fp:
			writer = csv.writer(fp, delimiter=str(','))
			for item in self.label_list: 
				writer.writerow(item)

		tkinter.messagebox.showinfo('导出成功到:out_label.csv')

	def load_image(self, idx):
		
		if idx >= len(self.img_list):
			return
		self.label_idx.configure(text=str(idx))

		item = self.img_list[idx]
		img = cv.imread("images//" + item[0])
		cv.namedWindow("image")
		cv.imshow("image", img)

		l_objs = item[1].split(';')
		for i in range(len(l_objs)):
			rect = Rect(i, l_objs[i])
			rect.draw_me(img)
		cv.imshow("image", img)
		cv.waitKey(20)

	def next_pic(self):
		if self.cur_idx >=len(self.img_list)-1:
			tkinter.messagebox.showinfo('没有了')
			return
		self.cur_idx += 1
		self.ent_label.delete(0, 'end')

		self.load_image(self.cur_idx)

	def prev_pic(self):
		if self.cur_idx <= 0:
			tkinter.messagebox.showinfo('到头了')
			return
		self.cur_idx -= 1
		self.ent_label.delete(0, 'end')

		self.load_image(self.cur_idx)

	def submit_label(self):
		
		item = [self.img_list[self.cur_idx][0],self.img_list[self.cur_idx][1], 
				self.ent_label.get()]
		self.label_list.append(item)

app = App()

