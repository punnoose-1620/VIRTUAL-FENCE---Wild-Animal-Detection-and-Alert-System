from tkinter import *
from tkinter import messagebox
import tkinter as tk
from tkinter.ttk import *
from tkinter import ttk
from PIL import Image, ImageTk
from tkinter import filedialog
import xlrd
from tkinter import Frame
import cv2
import numpy as np
import serial
from keras.models import Model
import tensorflow as tf
from keras.layers import Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.merge import concatenate
import requests
from keras.applications.mobilenet import MobileNet
from keras.applications import InceptionV3
#from yoloutils import *


buzzert = 0
buzzere = 0
window = tk.Tk()  # Creating label for  tkinter window
window.title("VIRTUAL FENCE")
window.geometry('600x600')
label_1 = Label(window, font=('italics', 12, 'bold'), text="VIRTUAL FENCE").pack()


def resize_image(event):  # Creating maximize and minimize option for background image
    new_width = event.width
    new_height = event.height
    image = copy_of_image.resize((new_width, new_height))
    photo = ImageTk.PhotoImage(image)
    label.config(image=photo)
    label.image = photo


image = Image.open('image.jpg')  # Loading my image of Background
copy_of_image = image.copy()
photo = ImageTk.PhotoImage(image)
label = ttk.Label(window, image=photo)
label.bind('<Configure>', resize_image)
label.pack(fill=BOTH, expand=YES)
style = Style()

style.configure('W.TButton', font =
               ('calibri', 10, 'bold'),
                foreground = 'green',bg='red')
explanation="Wild animal detection and alert system"
w = tk.Label(window,text=explanation,font=('Tempus Sans ITC', 13, 'bold')).place(x=135,y=310)

button_1 = Button(window, text = 'Open',width=10,
                  style = 'W.TButton',
             command = lambda:new_page()).place(x=250, y=350)


def new_page():
    window.destroy()

    root = tk.Tk()
    root.title("VIRTUAL FENCE")

    root.geometry('600x600')
    image = Image.open('green.jpg')
    photo_image = ImageTk.PhotoImage(image)
    label = tk.Label(root, image=photo_image)
    label.pack()
    style = Style()
    style.configure('W.TButton', font=
            ('calibri', 10, 'bold'),
                    foreground='green')
    Button(root, text='Open video', width=25,
                      style='W.TButton',
                      command=lambda: browsefunc()).place(x=210, y=100)
    frame1 = Frame(root, highlightbackground="green", highlightcolor="red", highlightthickness=1, width=200,
                   height=10, bd=0)
    frame1.place(x=200, y=160)
    Button(root, text='Start Detection', width=25,
           style='W.TButton',
           command=lambda: checkingfile()).place(x=210, y=200)

    def browsefunc():
        global file

        file = 1
        file = filedialog.askopenfilename(initialdir="C:/Users/Sharooque/Desktop/videos", title="Select file",
                                          filetypes=(("mp4 files", "*.mp4"), ("all files", "*.*")))
        file_1 = file.replace('/', "\\\\")
        print(file_1)



    root.resizable(0, 0)

    def checkingfile():
        global name
        if file is not None:

           messagebox.showinfo("message", "Detection running")

           ser = serial.Serial('com5', 9600)

           input_size = 224
           max_box_per_image = 10
           labels = ["n02398521", "n03633091", "n03188531", "n07742313", "n02009912", "n04483307", "n01748264",
                     "n01614925",
                     "n02028035", "n04468005", "n02087394", "n04530566", "n02094114", "n03777568", "n04146614",
                     "n02123045",
                     "n02098286", "n02025239", "n03314780", "n03124170", "n04591713", "n00477639", "n04542943",
                     "n02403003",
                     "n02113978", "n03062245", "n02958343", "n01616318", "n04019541", "n03690938", "n03498962",
                     "n03467517",
                     "n07739125", "n07718472", "n03792782", "n02883205", "n02108422", "n02324045", "n03950228",
                     "n03764736",
                     "n04118538", "n07695742", "n02951585", "n04336792", "n04554684", "n02395406", "n02276258",
                     "n02085936",
                     "n02120505", "n03120491", "n02007558", "n04070727", "n04147183", "n04376876", "n02108551",
                     "n02002556",
                     "n08256735", "n02107142", "n04065464", "n02493793", "n03001627", "n03991062", "n03447447",
                     "n02094258",
                     "n02326432", "n02444819", "n02086646", "n02356798", "n03483316", "n02105412", "n02121808",
                     "n04536866",
                     "n04356056", "n01828970", "n02098413", "n02107683", "n07718747", "n01820546", "n02787622",
                     "n01737021",
                     "n02992211", "n02693246", "n04540053", "n01855032", "n02124075", "n03745146", "n02017213",
                     "n01682714",
                     "n03958227", "n02503517", "n02091134", "n02687172", "n02096437", "n02422106", "n02489166",
                     "n02011460",
                     "n02814533", "n03535780", "n02012849", "n02807133", "n01665541", "n02091244", "n02924116",
                     "n03770679",
                     "n03179701", "n03452741", "n07714571", "n03995372", "n02132136", "n04154565", "n03961711",
                     "n03804744",
                     "n01694178", "n02110806", "n02930766", "n04429376", "n03125729", "n02110341", "n01518878",
                     "n04487394",
                     "n03344393", "n01795545", "n02095314", "n07753113", "n02107574", "n04606251", "n01776313",
                     "n02107312",
                     "n01728572", "n02099601", "n02412080", "n02870880", "n02006656", "n07693725", "n02101556",
                     "n02088632",
                     "n02979186", "n02086910", "n02098105", "n02777292", "n04259630", "n02105505", "n03770439",
                     "n03814639",
                     "n04517823", "n02094433", "n04228054", "n02088364", "n01534433", "n03372029", "n01592084",
                     "n02672831",
                     "n02095570", "n02112137", "n02764044", "n02108000", "n04591157", "n01797886", "n01558993",
                     "n02374451",
                     "n02268443", "n01729322", "n08539072", "n04026417", "n02095889", "n02113712", "n01675722",
                     "n04557648",
                     "n03196217", "n02110627", "n01829413", "n02981792", "n04141076", "n02131653", "n07930864",
                     "n02129165",
                     "n02119022", "n02096051", "n02129604", "n02437136", "n01742172", "n03476991", "n01734418",
                     "n02037110",
                     "n03481172", "n01601694", "n01735189", "n03619890", "n03761084", "n03109150", "n02486261",
                     "n02102480",
                     "n06892775", "n02092339", "n03891251", "n02093428", "n04074963", "n02509815", "n07880968",
                     "n02835271",
                     "n02089973", "n01608432", "n03400231", "n02091467", "n03782006", "n02097130", "n04285008",
                     "n01817953",
                     "n02815834", "n02009229", "n04037443", "n01532829", "n01667114", "n02701002", "n02488702",
                     "n03494278",
                     "n02799071", "n04332243", "n10148035", "n02206856", "n01744401", "n03642806", "n02397096",
                     "n03841666",
                     "n03769881", "n02093754", "n02123597", "n02325366", "n07697100", "n02102040", "n02486410",
                     "n02018207",
                     "n04116512", "n02454379", "n01843065", "n01882714", "n02391049", "n01990800", "n07697537",
                     "n02279972",
                     "n03529860", "n02051845", "n04131690", "n01784675", "n04256520", "n07697313", "n07734744",
                     "n02280649",
                     "n02791124", "n02088094", "n04209133", "n01729977", "n02071294", "n03100240", "n04357314",
                     "n07749582",
                     "n01739381", "n02104365", "n03673027", "n02100735", "n02097474", "n02951358", "n02437312",
                     "n02277742",
                     "n04612504", "n02033041", "n04311174", "n02109047", "n02691156", "n02096585", "n03063599",
                     "n02104029",
                     "n02504458", "n00007846", "n02102177", "n02317335", "n04380533", "n04263257", "n01503061",
                     "n01855672",
                     "n03017168", "n02056570", "n02097209", "n02346627", "n03942813", "n02101006", "n02396427",
                     "n07583066",
                     "n02119789", "n03141823", "n02097298", "n01944390", "n02076196", "n01753488", "n01749939",
                     "n08517676",
                     "n01498041", "n03594945", "n03495258", "n02091635", "n02786058", "n02013706", "n02111500",
                     "n02492660",
                     "n04335209", "n01537544", "n02062744", "n03670208", "n04334599", "n04273569", "n03908618",
                     "n02690373",
                     "n02108915", "n07745940", "n03599486", "n03272010", "n01860187", "n02084071", "n02892767",
                     "n02099429",
                     "n02113624", "n03581125", "n09835506", "n02109961", "n03038685", "n01847000", "n02834778",
                     "n02484322",
                     "n04039381", "n02097658", "n02165456", "n01818515", "n01669191", "n02766320", "n01693334",
                     "n03085013",
                     "n01687978", "n02493509", "n01807496", "n02105641", "n01806143", "n01798484", "n02088466",
                     "n03785016",
                     "n02110958", "n02113186", "n07768694", "n03376595", "n02090379", "n07747607", "n01530575",
                     "n04515003",
                     "n03868242", "n03832673", "n07753275", "n02504013", "n02058221", "n03297495", "n02490219",
                     "n02102318",
                     "n02106662", "n06874185", "n01728920", "n02879718", "n02066245", "n01677366", "n02102973",
                     "n04270147",
                     "n01726692", "n01695060", "n02088238", "n03337140", "n03394916", "n03445777", "n01819313",
                     "n02099712",
                     "n01910747", "n02090622", "n03120778", "n02096177", "n02105056", "n08518171", "n02402425",
                     "n03916031",
                     "n07753592", "n04347754", "n02395003", "n02089078", "n02134084", "n02419796", "n02027492",
                     "n01982650",
                     "n02100236", "n02111129", "n04330267", "n02090721", "n04371430", "n02018795", "n04004767",
                     "n02101388",
                     "n02488291", "n03797390", "n02112018", "n01496331", "n02676566", "n02487347", "n02112350",
                     "n03134739",
                     "n01806567", "n01674464", "n04254680", "n03710721", "n02389026", "n03584254", "n02106166",
                     "n01740131",
                     "n04081281", "n02106550", "n02093256", "n07873807", "n02133161", "n02917067", "n01692333",
                     "n03538406",
                     "n03928116", "n03127747", "n01667778", "n02113023", "n03110669", "n03791053", "n03207941",
                     "n01580077",
                     "n01688243", "n01770393", "n02492035", "n03063338", "n02422699", "n02106030", "n03128519",
                     "n02108089",
                     "n04509417", "n01514668", "n04392985", "n03255030", "n02423022", "n04409515", "n02219486",
                     "n02100877",
                     "n02110063", "n02096294", "n02091032", "n02099849", "n02821627", "n02085620", "n03838899",
                     "n03676483",
                     "n03271574", "n01664065", "n01983481", "n02077923", "n02970849", "n02484975", "n01641577",
                     "n02120079",
                     "n02769748", "n03211117", "n03379051", "n02134418", "n01751748", "n02839592", "n01824575",
                     "n01560419",
                     "n03249569", "n00141669", "n03790512", "n02828884", "n03759954", "n02111277", "n02110185",
                     "n03775546",
                     "n03793489", "n02099267", "n02113799", "n02112706", "n01639765", "n01984695", "n02109525",
                     "n03720891",
                     "n01531178", "n03131574", "n02804414", "n01755581", "n04442312", "n03982430", "n07615774",
                     "n02123394",
                     "n02105855", "n04152593", "n03513137", "n02281787", "n02091831", "n02328150", "n01843383",
                     "n01833805",
                     "n04252225", "n02105162", "n02118333", "n02692232", "n02002724", "n02355227", "n02086240",
                     "n02100583",
                     "n02342885", "n02097047", "n02445715", "n01443537", "n03947888", "n02106382", "n04317175",
                     "n02411705",
                     "n01514859", "n02092002", "n02111889", "n02281406", "n10565667", "n04023962", "n03000684",
                     "n04118776",
                     "n01685808", "n01796340", "n02093991", "n01622779", "n03416640", "n02087046", "n02687992",
                     "n02093859",
                     "n02802426", "n04344873", "n04379243", "n03445924", "n02089867", "n01756291", "n02085782",
                     "n02415577",
                     "n03908714", "n01644373", "n01582220", "n02494079", "n04254120", "n02840245", "n02107908",
                     "n04099969",
                     "n01644900", "n02093647", "n03095699", "n02105251", "n02880940", "n01689811", "n02510455",
                     "n01662784",
                     "n01495701", "n03201208", "n04487081", "n03636649", "n03662601", "n02274259", "n04252077",
                     "n07720875",
                     "n02086079"]
           nb_class = len(labels)
           nb_box = 5
           class_wt = np.ones(nb_class, dtype='float32')
           anchors = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]

           class BoundBox:
               def __init__(self, x, y, w, h, c=None, classes=None):
                   self.x = x
                   self.y = y
                   self.w = w
                   self.h = h

                   self.c = c
                   self.classes = classes

                   self.label = -1
                   self.score = -1

               def get_label(self):
                   if self.label == -1:
                       self.label = np.argmax(self.classes)

                   return self.label

               def get_score(self):
                   if self.score == -1:
                       self.score = self.classes[self.get_label()]

                   return self.score

           def normalize(image):
               return image / 255.

           def draw_boxes(image, boxes, labels):
               for box in boxes:
                   xmin = int((box.x - box.w / 2) * image.shape[1])
                   xmax = int((box.x + box.w / 2) * image.shape[1])
                   ymin = int((box.y - box.h / 2) * image.shape[0])
                   ymax = int((box.y + box.h / 2) * image.shape[0])

                   cid = labels[box.get_label()]
                   global buzzert
                   global buzzere

                   if cid == "n02129604" or cid == 'n02391049' or cid =='n02129165':
                       # printing("Tiger")
                       buzzert = 1
                       cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 0), 3)
                       cv2.putText(image,
                                   "Tiger" + ' ' + str(box.get_score()),
                                   (xmin, ymin - 13),
                                   cv2.FONT_HERSHEY_COMPLEX,
                                   1e-3 * image.shape[0],
                                   (0, 0, 0), 2)
                       if buzzert == 1:
                           ser.write('1'.encode())

                   if cid == "n02504458" or cid == "n02504013" or cid == "n02437136":
                       # printing("Elephant")
                       buzzere = 1
                       cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 0), 3)
                       cv2.putText(image,
                                   "Elephant" + ' ' + str(box.get_score()),
                                   (xmin, ymin - 13),
                                   cv2.FONT_HERSHEY_COMPLEX,
                                   1e-3 * image.shape[0],
                                   (0, 0, 0), 2)
                       if buzzere == 1:
                           ser.write('1'.encode())

               return image

           def interval_overlap(interval_a, interval_b):
               x1, x2 = interval_a
               x3, x4 = interval_b

               if x3 < x1:
                   if x4 < x1:
                       return 0
                   else:
                       return min(x2, x4) - x1
               else:
                   if x2 < x3:
                       return 0
                   else:
                       return min(x2, x4) - x3

           def bbox_iou(box1, box2):
               x1_min = box1.x - box1.w / 2
               x1_max = box1.x + box1.w / 2
               y1_min = box1.y - box1.h / 2
               y1_max = box1.y + box1.h / 2

               x2_min = box2.x - box2.w / 2
               x2_max = box2.x + box2.w / 2
               y2_min = box2.y - box2.h / 2
               y2_max = box2.y + box2.h / 2

               intersect_w = interval_overlap([x1_min, x1_max], [x2_min, x2_max])
               intersect_h = interval_overlap([y1_min, y1_max], [y2_min, y2_max])

               intersect = intersect_w * intersect_h

               union = box1.w * box1.h + box2.w * box2.h - intersect

               return float(intersect) / union

           def sigmoid(x):
               return 1. / (1. + np.exp(-x))

           def softmax(x, axis=-1, t=-100.):
               x = x - np.max(x)
               if np.min(x) < t:
                   x = x / np.min(x) * t
               e_x = np.exp(x)
               return e_x / e_x.sum(axis, keepdims=True)

           def decode_netout(netout, obj_threshold=0.3, nms_threshold=0.3):
               grid_h, grid_w, nb_box = netout.shape[:3]

               boxes = []

               # decode the output by the network
               netout[..., 4] = sigmoid(netout[..., 4])
               netout[..., 5:] = netout[..., 4][..., np.newaxis] * softmax(netout[..., 5:])
               netout[..., 5:] *= netout[..., 5:] > obj_threshold

               for row in range(grid_h):
                   for col in range(grid_w):
                       for b in range(nb_box):
                           # from 4th element onwards are confidence and class classes
                           classes = netout[row, col, b, 5:]

                           if np.sum(classes) > 0:
                               # first 4 elements are x, y, w, and h
                               x, y, w, h = netout[row, col, b, :4]

                               x = (col + sigmoid(x)) / grid_w  # center position, unit: image width
                               y = (row + sigmoid(y)) / grid_h  # center position, unit: image height
                               w = anchors[2 * b + 0] * np.exp(w) / grid_w  # unit: image width
                               h = anchors[2 * b + 1] * np.exp(h) / grid_h  # unit: image height
                               confidence = netout[row, col, b, 4]

                               box = BoundBox(x, y, w, h, confidence, classes)

                               boxes.append(box)

               # suppress non-maximal boxes
               for c in range(nb_class):
                   sorted_indices = list(reversed(np.argsort([box.classes[c] for box in boxes])))

                   for i in range(len(sorted_indices)):
                       index_i = sorted_indices[i]

                       if boxes[index_i].classes[c] == 0:
                           continue
                       else:
                           for j in range(i + 1, len(sorted_indices)):
                               index_j = sorted_indices[j]

                               if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_threshold:
                                   boxes[index_j].classes[c] = 0

               # remove the boxes which are less likely than a obj_threshold
               boxes = [box for box in boxes if box.get_score() > obj_threshold]

               return boxes

           input_size = 224
           max_box_per_image = 10
           labels = ["n02398521", "n03633091", "n03188531", "n07742313", "n02009912", "n04483307", "n01748264",
                     "n01614925", "n02028035", "n04468005", "n02087394", "n04530566", "n02094114", "n03777568",
                     "n04146614", "n02123045", "n02098286", "n02025239", "n03314780", "n03124170", "n04591713",
                     "n00477639", "n04542943", "n02403003", "n02113978", "n03062245", "n02958343", "n01616318",
                     "n04019541", "n03690938", "n03498962", "n03467517", "n07739125", "n07718472", "n03792782",
                     "n02883205", "n02108422", "n02324045", "n03950228", "n03764736", "n04118538", "n07695742",
                     "n02951585", "n04336792", "n04554684", "n02395406", "n02276258", "n02085936", "n02120505",
                     "n03120491", "n02007558", "n04070727", "n04147183", "n04376876", "n02108551", "n02002556",
                     "n08256735", "n02107142", "n04065464", "n02493793", "n03001627", "n03991062", "n03447447",
                     "n02094258", "n02326432", "n02444819", "n02086646", "n02356798", "n03483316", "n02105412",
                     "n02121808", "n04536866", "n04356056", "n01828970", "n02098413", "n02107683", "n07718747",
                     "n01820546", "n02787622", "n01737021", "n02992211", "n02693246", "n04540053", "n01855032",
                     "n02124075", "n03745146", "n02017213", "n01682714", "n03958227", "n02503517", "n02091134",
                     "n02687172", "n02096437", "n02422106", "n02489166", "n02011460", "n02814533", "n03535780",
                     "n02012849", "n02807133", "n01665541", "n02091244", "n02924116", "n03770679", "n03179701",
                     "n03452741", "n07714571", "n03995372", "n02132136", "n04154565", "n03961711", "n03804744",
                     "n01694178", "n02110806", "n02930766", "n04429376", "n03125729", "n02110341", "n01518878",
                     "n04487394", "n03344393", "n01795545", "n02095314", "n07753113", "n02107574", "n04606251",
                     "n01776313", "n02107312", "n01728572", "n02099601", "n02412080", "n02870880", "n02006656",
                     "n07693725", "n02101556", "n02088632", "n02979186", "n02086910", "n02098105", "n02777292",
                     "n04259630", "n02105505", "n03770439", "n03814639", "n04517823", "n02094433", "n04228054",
                     "n02088364", "n01534433", "n03372029", "n01592084", "n02672831", "n02095570", "n02112137",
                     "n02764044", "n02108000", "n04591157", "n01797886", "n01558993", "n02374451", "n02268443",
                     "n01729322", "n08539072", "n04026417", "n02095889", "n02113712", "n01675722", "n04557648",
                     "n03196217", "n02110627", "n01829413", "n02981792", "n04141076", "n02131653", "n07930864",
                     "n02129165", "n02119022", "n02096051", "n02129604", "n02437136", "n01742172", "n03476991",
                     "n01734418", "n02037110", "n03481172", "n01601694", "n01735189", "n03619890", "n03761084",
                     "n03109150", "n02486261", "n02102480", "n06892775", "n02092339", "n03891251", "n02093428",
                     "n04074963", "n02509815", "n07880968", "n02835271", "n02089973", "n01608432", "n03400231",
                     "n02091467", "n03782006", "n02097130", "n04285008", "n01817953", "n02815834", "n02009229",
                     "n04037443", "n01532829", "n01667114", "n02701002", "n02488702", "n03494278", "n02799071",
                     "n04332243", "n10148035", "n02206856", "n01744401", "n03642806", "n02397096", "n03841666",
                     "n03769881", "n02093754", "n02123597", "n02325366", "n07697100", "n02102040", "n02486410",
                     "n02018207", "n04116512", "n02454379", "n01843065", "n01882714", "n02391049", "n01990800",
                     "n07697537", "n02279972", "n03529860", "n02051845", "n04131690", "n01784675", "n04256520",
                     "n07697313", "n07734744", "n02280649", "n02791124", "n02088094", "n04209133", "n01729977",
                     "n02071294", "n03100240", "n04357314", "n07749582", "n01739381", "n02104365", "n03673027",
                     "n02100735", "n02097474", "n02951358", "n02437312", "n02277742", "n04612504", "n02033041",
                     "n04311174", "n02109047", "n02691156", "n02096585", "n03063599", "n02104029", "n02504458",
                     "n00007846", "n02102177", "n02317335", "n04380533", "n04263257", "n01503061", "n01855672",
                     "n03017168", "n02056570", "n02097209", "n02346627", "n03942813", "n02101006", "n02396427",
                     "n07583066", "n02119789", "n03141823", "n02097298", "n01944390", "n02076196", "n01753488",
                     "n01749939", "n08517676", "n01498041", "n03594945", "n03495258", "n02091635", "n02786058",
                     "n02013706", "n02111500", "n02492660", "n04335209", "n01537544", "n02062744", "n03670208",
                     "n04334599", "n04273569", "n03908618", "n02690373", "n02108915", "n07745940", "n03599486",
                     "n03272010", "n01860187", "n02084071", "n02892767", "n02099429", "n02113624", "n03581125",
                     "n09835506", "n02109961", "n03038685", "n01847000", "n02834778", "n02484322", "n04039381",
                     "n02097658", "n02165456", "n01818515", "n01669191", "n02766320", "n01693334", "n03085013",
                     "n01687978", "n02493509", "n01807496", "n02105641", "n01806143", "n01798484", "n02088466",
                     "n03785016", "n02110958", "n02113186", "n07768694", "n03376595", "n02090379", "n07747607",
                     "n01530575", "n04515003", "n03868242", "n03832673", "n07753275", "n02504013", "n02058221",
                     "n03297495", "n02490219", "n02102318", "n02106662", "n06874185", "n01728920", "n02879718",
                     "n02066245", "n01677366", "n02102973", "n04270147", "n01726692", "n01695060", "n02088238",
                     "n03337140", "n03394916", "n03445777", "n01819313", "n02099712", "n01910747", "n02090622",
                     "n03120778", "n02096177", "n02105056", "n08518171", "n02402425", "n03916031", "n07753592",
                     "n04347754", "n02395003", "n02089078", "n02134084", "n02419796", "n02027492", "n01982650",
                     "n02100236", "n02111129", "n04330267", "n02090721", "n04371430", "n02018795", "n04004767",
                     "n02101388", "n02488291", "n03797390", "n02112018", "n01496331", "n02676566", "n02487347",
                     "n02112350", "n03134739", "n01806567", "n01674464", "n04254680", "n03710721", "n02389026",
                     "n03584254", "n02106166", "n01740131", "n04081281", "n02106550", "n02093256", "n07873807",
                     "n02133161", "n02917067", "n01692333", "n03538406", "n03928116", "n03127747", "n01667778",
                     "n02113023", "n03110669", "n03791053", "n03207941", "n01580077", "n01688243", "n01770393",
                     "n02492035", "n03063338", "n02422699", "n02106030", "n03128519", "n02108089", "n04509417",
                     "n01514668", "n04392985", "n03255030", "n02423022", "n04409515", "n02219486", "n02100877",
                     "n02110063", "n02096294", "n02091032", "n02099849", "n02821627", "n02085620", "n03838899",
                     "n03676483", "n03271574", "n01664065", "n01983481", "n02077923", "n02970849", "n02484975",
                     "n01641577", "n02120079", "n02769748", "n03211117", "n03379051", "n02134418", "n01751748",
                     "n02839592", "n01824575", "n01560419", "n03249569", "n00141669", "n03790512", "n02828884",
                     "n03759954", "n02111277", "n02110185", "n03775546", "n03793489", "n02099267", "n02113799",
                     "n02112706", "n01639765", "n01984695", "n02109525", "n03720891", "n01531178", "n03131574",
                     "n02804414", "n01755581", "n04442312", "n03982430", "n07615774", "n02123394", "n02105855",
                     "n04152593", "n03513137", "n02281787", "n02091831", "n02328150", "n01843383", "n01833805",
                     "n04252225", "n02105162", "n02118333", "n02692232", "n02002724", "n02355227", "n02086240",
                     "n02100583", "n02342885", "n02097047", "n02445715", "n01443537", "n03947888", "n02106382",
                     "n04317175", "n02411705", "n01514859", "n02092002", "n02111889", "n02281406", "n10565667",
                     "n04023962", "n03000684", "n04118776", "n01685808", "n01796340", "n02093991", "n01622779",
                     "n03416640", "n02087046", "n02687992", "n02093859", "n02802426", "n04344873", "n04379243",
                     "n03445924", "n02089867", "n01756291", "n02085782", "n02415577", "n03908714", "n01644373",
                     "n01582220", "n02494079", "n04254120", "n02840245", "n02107908", "n04099969", "n01644900",
                     "n02093647", "n03095699", "n02105251", "n02880940", "n01689811", "n02510455", "n01662784",
                     "n01495701", "n03201208", "n04487081", "n03636649", "n03662601", "n02274259", "n04252077",
                     "n07720875", "n02086079"]
           nb_class = len(labels)
           nb_box = 5
           class_wt = np.ones(nb_class, dtype='float32')
           anchors = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]

           # the function to implement the orgnization layer (thanks to github.com/allanzelener/YAD2K)
           def space_to_depth_x2(x):
               return tf.space_to_depth(x, block_size=2)

           def predict(model, image):
               image = cv2.resize(image, (input_size, input_size))
               image = normalize(image)

               input_image = image[:, :, ::-1]
               input_image = np.expand_dims(input_image, 0)
               dummy_array = dummy_array = np.zeros((1, 1, 1, 1, max_box_per_image, 4))

               netout = model.predict([input_image, dummy_array])[0]
               boxes = decode_netout(netout)
               return boxes

           def YOLOMODEL(path):
               input_image = Input(shape=(input_size, input_size, 3))
               true_boxes = Input(shape=(1, 1, 1, max_box_per_image, 4))

               # Layer 1
               x = Conv2D(32, (3, 3), strides=(1, 1), padding='same', name='conv_1', use_bias=False)(input_image)
               x = BatchNormalization(name='norm_1')(x)
               x = LeakyReLU(alpha=0.1)(x)
               x = MaxPooling2D(pool_size=(2, 2))(x)

               # Layer 2
               x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', name='conv_2', use_bias=False)(x)
               x = BatchNormalization(name='norm_2')(x)
               x = LeakyReLU(alpha=0.1)(x)
               x = MaxPooling2D(pool_size=(2, 2))(x)

               # Layer 3
               x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', name='conv_3', use_bias=False)(x)
               x = BatchNormalization(name='norm_3')(x)
               x = LeakyReLU(alpha=0.1)(x)

               # Layer 4
               x = Conv2D(64, (1, 1), strides=(1, 1), padding='same', name='conv_4', use_bias=False)(x)
               x = BatchNormalization(name='norm_4')(x)
               x = LeakyReLU(alpha=0.1)(x)

               # Layer 5
               x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', name='conv_5', use_bias=False)(x)
               x = BatchNormalization(name='norm_5')(x)
               x = LeakyReLU(alpha=0.1)(x)
               x = MaxPooling2D(pool_size=(2, 2))(x)

               # Layer 6
               x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='conv_6', use_bias=False)(x)
               x = BatchNormalization(name='norm_6')(x)
               x = LeakyReLU(alpha=0.1)(x)

               # Layer 7
               x = Conv2D(128, (1, 1), strides=(1, 1), padding='same', name='conv_7', use_bias=False)(x)
               x = BatchNormalization(name='norm_7')(x)
               x = LeakyReLU(alpha=0.1)(x)

               # Layer 8
               x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='conv_8', use_bias=False)(x)
               x = BatchNormalization(name='norm_8')(x)
               x = LeakyReLU(alpha=0.1)(x)
               x = MaxPooling2D(pool_size=(2, 2))(x)

               # Layer 9
               x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv_9', use_bias=False)(x)
               x = BatchNormalization(name='norm_9')(x)
               x = LeakyReLU(alpha=0.1)(x)

               # Layer 10
               x = Conv2D(256, (1, 1), strides=(1, 1), padding='same', name='conv_10', use_bias=False)(x)
               x = BatchNormalization(name='norm_10')(x)
               x = LeakyReLU(alpha=0.1)(x)

               # Layer 11
               x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv_11', use_bias=False)(x)
               x = BatchNormalization(name='norm_11')(x)
               x = LeakyReLU(alpha=0.1)(x)

               # Layer 12
               x = Conv2D(256, (1, 1), strides=(1, 1), padding='same', name='conv_12', use_bias=False)(x)
               x = BatchNormalization(name='norm_12')(x)
               x = LeakyReLU(alpha=0.1)(x)

               # Layer 13
               x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv_13', use_bias=False)(x)
               x = BatchNormalization(name='norm_13')(x)
               x = LeakyReLU(alpha=0.1)(x)

               skip_connection = x

               x = MaxPooling2D(pool_size=(2, 2))(x)

               # Layer 14
               x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_14', use_bias=False)(x)
               x = BatchNormalization(name='norm_14')(x)
               x = LeakyReLU(alpha=0.1)(x)

               # Layer 15
               x = Conv2D(512, (1, 1), strides=(1, 1), padding='same', name='conv_15', use_bias=False)(x)
               x = BatchNormalization(name='norm_15')(x)
               x = LeakyReLU(alpha=0.1)(x)

               # Layer 16
               x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_16', use_bias=False)(x)
               x = BatchNormalization(name='norm_16')(x)
               x = LeakyReLU(alpha=0.1)(x)

               # Layer 17
               x = Conv2D(512, (1, 1), strides=(1, 1), padding='same', name='conv_17', use_bias=False)(x)
               x = BatchNormalization(name='norm_17')(x)
               x = LeakyReLU(alpha=0.1)(x)

               # Layer 18
               x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_18', use_bias=False)(x)
               x = BatchNormalization(name='norm_18')(x)
               x = LeakyReLU(alpha=0.1)(x)

               # Layer 19
               x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_19', use_bias=False)(x)
               x = BatchNormalization(name='norm_19')(x)
               x = LeakyReLU(alpha=0.1)(x)

               # Layer 20
               x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_20', use_bias=False)(x)
               x = BatchNormalization(name='norm_20')(x)
               x = LeakyReLU(alpha=0.1)(x)

               # Layer 21
               skip_connection = Conv2D(64, (1, 1), strides=(1, 1), padding='same', name='conv_21', use_bias=False)(
                   skip_connection)
               skip_connection = BatchNormalization(name='norm_21')(skip_connection)
               skip_connection = LeakyReLU(alpha=0.1)(skip_connection)
               skip_connection = Lambda(space_to_depth_x2)(skip_connection)

               x = concatenate([skip_connection, x])

               # Layer 22
               x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_22', use_bias=False)(x)
               x = BatchNormalization(name='norm_22')(x)
               x = LeakyReLU(alpha=0.1)(x)
               feature_extractor = Model(input_image, x, name='FULLYOLO')
               features = feature_extractor(input_image)

               grid_h, grid_w = feature_extractor.get_output_shape_at(-1)[1:3]

               # make the object detection layer
               output = Conv2D(nb_box * (4 + 1 + nb_class),
                               (1, 1), strides=(1, 1),
                               padding='same',
                               name='conv_23',
                               kernel_initializer='lecun_normal')(features)
               output = Reshape((grid_h, grid_w, nb_box, 4 + 1 + nb_class))(output)
               output = Lambda(lambda args: args[0])([output, true_boxes])

               model = Model([input_image, true_boxes], output)

               # initialize the weights of the detection layer
               layer = model.layers[-4]
               weights = layer.get_weights()

               new_kernel = np.random.normal(size=weights[0].shape) / (grid_h * grid_w)
               new_bias = np.random.normal(size=weights[1].shape) / (grid_h * grid_w)

               layer.set_weights([new_kernel, new_bias])

               model.load_weights(path)

               #print(model.summary())
               return model

           def __main__():

               path = 'bestModel_backup.h5'
               yolo = YOLOMODEL(path)

               #   Predict bounding boxes
               counter = 0
               cap = cv2.VideoCapture(file)
               i = 0
               while True:
                   ret, frame = cap.read()
                   if not ret:
                       break
                   if counter == 0:
                       if buzzert == 1:
                           url = "https://www.fast2sms.com/dev/bulk"
                           payload = "sender_id=FSTSMS&message=ALERT : TIGER CROSSED BOUNDARY.&language=english&route=p&numbers=7012879096"
                           headers = {
                               'authorization': "mVUXHstOqgn8T9I1GieKdfw4LvQByhj3Ro2PzAl0YaZMD6NJb7ps30QDIXfK2VzEgqj9USerOTomhnC7",
                               'Content-Type': "application/x-www-form-urlencoded",
                               'Cache-Control': "no-cache",
                           }
                           response = requests.request("POST", url, data=payload, headers=headers)
                           print(response.text)
                           counter = 1

                       if buzzere == 1:
                           url = "https://www.fast2sms.com/dev/bulk"
                           payload = "sender_id=FSTSMS&message=ALERT : ELEPHANT CROSSED BOUNDARY.&language=english&route=p&numbers=7012879096"
                           headers = {
                               'authorization': "mVUXHstOqgn8T9I1GieKdfw4LvQByhj3Ro2PzAl0YaZMD6NJb7ps30QDIXfK2VzEgqj9USerOTomhnC7",
                               'Content-Type': "application/x-www-form-urlencoded",
                               'Cache-Control': "no-cache",
                           }
                           response = requests.request("POST", url, data=payload, headers=headers)
                           print(response.text)
                           counter = 1
                   boxes = predict(yolo, frame)
                   frame = draw_boxes(frame, boxes, labels)
                   # cv2.imwrite('../tmp/'+str(i)+'.jpg', frame)
                   if cv2.waitKey(1) & 0xFF == ord('q'):
                       break
                   cv2.imshow('frame', frame)

                   i += 1

               cap.release()


           __main__()














           messagebox.showinfo("message", " Prediction is completed")
           root.destroy()











    root.mainloop()


window.resizable(0, 0)
window.mainloop()
