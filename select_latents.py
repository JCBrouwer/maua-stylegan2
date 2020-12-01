import gc, math
import argparse
import tkinter as tk
import numpy as np
from PIL import Image, ImageTk
import torch as th
import torch.nn.functional as F
import torchvision
from models.stylegan2 import Generator as G_style2
import tkinter as tk

# --- classes ---
try:
    from Tkinter import Canvas, Frame
    from ttk import Scrollbar

    from Tkconstants import *
except ImportError:
    from tkinter import Canvas, Frame
    from tkinter.ttk import Scrollbar

    from tkinter.constants import *

import platform

OS = platform.system()


class HoverButton(tk.Button):
    def __init__(self, master, **kw):
        tk.Button.__init__(self, master=master, **kw)
        self.defaultBackground = self["background"]
        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)

    def on_enter(self, e):
        self["background"] = self["activebackground"]

    def on_leave(self, e):
        self["background"] = self.defaultBackground


class InvisibleScrollbar(Scrollbar):
    def set(self, lo, hi):
        self.tk.call("grid", "remove", self)
        Scrollbar.set(self, lo, hi)


class Mousewheel_Support(object):
    # implemetation of singleton pattern
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = object.__new__(cls)
        return cls._instance

    def __init__(self, root, horizontal_factor=1, vertical_factor=1):
        self._active_area = None
        if isinstance(horizontal_factor, int):
            self.horizontal_factor = horizontal_factor
        else:
            raise Exception("Vertical factor must be an integer.")
        if isinstance(vertical_factor, int):
            self.vertical_factor = vertical_factor
        else:
            raise Exception("Horizontal factor must be an integer.")
        if OS == "Linux":
            root.bind_all("<4>", self._on_mousewheel, add="+")
            root.bind_all("<5>", self._on_mousewheel, add="+")
        else:
            # Windows and MacOS
            root.bind_all("<MouseWheel>", self._on_mousewheel, add="+")

    def _on_mousewheel(self, event):
        if self._active_area:
            self._active_area.onMouseWheel(event)

    def _mousewheel_bind(self, widget):
        self._active_area = widget

    def _mousewheel_unbind(self):
        self._active_area = None

    def add_support_to(
        self, widget=None, xscrollbar=None, yscrollbar=None, what="units", horizontal_factor=None, vertical_factor=None
    ):
        if xscrollbar is None and yscrollbar is None:
            return
        if xscrollbar is not None:
            horizontal_factor = horizontal_factor or self.horizontal_factor
            xscrollbar.onMouseWheel = self._make_mouse_wheel_handler(widget, "x", self.horizontal_factor, what)
            xscrollbar.bind("<Enter>", lambda event, scrollbar=xscrollbar: self._mousewheel_bind(scrollbar))
            xscrollbar.bind("<Leave>", lambda event: self._mousewheel_unbind())
        if yscrollbar is not None:
            vertical_factor = vertical_factor or self.vertical_factor
            yscrollbar.onMouseWheel = self._make_mouse_wheel_handler(widget, "y", self.vertical_factor, what)
            yscrollbar.bind("<Enter>", lambda event, scrollbar=yscrollbar: self._mousewheel_bind(scrollbar))
            yscrollbar.bind("<Leave>", lambda event: self._mousewheel_unbind())
        main_scrollbar = yscrollbar if yscrollbar is not None else xscrollbar
        if widget is not None:
            if isinstance(widget, list) or isinstance(widget, tuple):
                list_of_widgets = widget
                for widget in list_of_widgets:
                    widget.bind("<Enter>", lambda event: self._mousewheel_bind(widget))
                    widget.bind("<Leave>", lambda event: self._mousewheel_unbind())
                    widget.onMouseWheel = main_scrollbar.onMouseWheel
            else:
                widget.bind("<Enter>", lambda event: self._mousewheel_bind(widget))
                widget.bind("<Leave>", lambda event: self._mousewheel_unbind())
                widget.onMouseWheel = main_scrollbar.onMouseWheel

    @staticmethod
    def _make_mouse_wheel_handler(widget, orient, factor=1 / 120, what="units"):
        view_command = getattr(widget, orient + "view")
        if OS == "Linux":

            def onMouseWheel(event):
                if event.num == 4:
                    view_command("scroll", (-1) * factor, what)
                elif event.num == 5:
                    view_command("scroll", factor, what)

        elif OS == "Windows":

            def onMouseWheel(event):
                view_command("scroll", (-1) * int((event.delta / 120) * factor), what)

        elif OS == "Darwin":

            def onMouseWheel(event):
                view_command("scroll", event.delta, what)

        return onMouseWheel


class Scrolling_Area(Frame, object):
    def __init__(
        self,
        master,
        width=None,
        anchor=N,
        height=None,
        mousewheel_speed=2,
        scroll_horizontally=True,
        xscrollbar=None,
        scroll_vertically=True,
        yscrollbar=None,
        background="black",
        inner_frame=Frame,
        **kw,
    ):
        Frame.__init__(self, master, class_="Scrolling_Area", background=background)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self._width = width
        self._height = height
        self.canvas = Canvas(self, background=background, highlightthickness=0, width=width, height=height)
        self.canvas.grid(row=0, column=0, sticky=N + E + W + S)
        if scroll_vertically:
            if yscrollbar is not None:
                self.yscrollbar = yscrollbar
            else:
                self.yscrollbar = InvisibleScrollbar(self, orient=VERTICAL)
                self.yscrollbar.grid(row=0, column=1, sticky=N + S)
            self.canvas.configure(yscrollcommand=self.yscrollbar.set)
            self.yscrollbar["command"] = self.canvas.yview
        else:
            self.yscrollbar = None
        if scroll_horizontally:
            if xscrollbar is not None:
                self.xscrollbar = xscrollbar
            else:
                self.xscrollbar = InvisibleScrollbar(self, orient=HORIZONTAL)
                self.xscrollbar.grid(row=1, column=0, sticky=E + W)
            self.canvas.configure(xscrollcommand=self.xscrollbar.set)
            self.xscrollbar["command"] = self.canvas.xview
        else:
            self.xscrollbar = None
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self.innerframe = inner_frame(self.canvas, **kw)
        self.innerframe.pack(anchor=anchor)
        self.canvas.create_window(0, 0, window=self.innerframe, anchor="nw", tags="inner_frame")
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        Mousewheel_Support(self).add_support_to(self.canvas, xscrollbar=self.xscrollbar, yscrollbar=self.yscrollbar)

    @property
    def width(self):
        return self.canvas.winfo_width()

    @width.setter
    def width(self, width):
        self.canvas.configure(width=width)

    @property
    def height(self):
        return self.canvas.winfo_height()

    @height.setter
    def height(self, height):
        self.canvas.configure(height=height)

    def set_size(self, width, height):
        self.canvas.configure(width=width, height=height)

    def _on_canvas_configure(self, event):
        width = max(self.innerframe.winfo_reqwidth(), event.width)
        height = max(self.innerframe.winfo_reqheight(), event.height)
        self.canvas.configure(scrollregion="0 0 %s %s" % (width, height))
        self.canvas.itemconfigure("inner_frame", width=width, height=height)

    def update_viewport(self):
        self.update()
        window_width = self.innerframe.winfo_reqwidth()
        window_height = self.innerframe.winfo_reqheight()
        if self._width is None:
            canvas_width = window_width
        else:
            canvas_width = min(self._width, window_width)
        if self._height is None:
            canvas_height = window_height
        else:
            canvas_height = min(self._height, window_height)
        self.canvas.configure(
            scrollregion="0 0 %s %s" % (window_width, window_height), width=self._width, height=self._height
        )
        self.canvas.itemconfigure("inner_frame", width=window_width, height=window_height)


th.set_grad_enabled(False)
th.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()

parser.add_argument("--ckpt", type=str)
parser.add_argument("--res", type=int, default=1024)
parser.add_argument("--truncation", type=int, default=1.5)
parser.add_argument("--noconst", action="store_false")

args = parser.parse_args()

name = args.ckpt.split("/")[-1].split(".")[0]
GENERATOR = (
    G_style2(size=args.res, style_dim=512, n_mlp=8, checkpoint=args.ckpt, output_size=1024, constant_input=args.noconst)
    .eval()
    .cuda()
)
# GENERATOR = G_style(checkpoint=args.ckpt, output_size=1024).eval().cuda()
# GENERATOR = th.nn.DataParallel(GENERATOR)

IMAGES_PER_ROW = 4
IMSIZE = (1920 - 240) // IMAGES_PER_ROW

ALL_LATENTS = []
DROP_IDXS = []
INTRO_IDXS = []
IMAGES = []


def generate_images(n):
    imgs = []
    for _ in range(n // 8):
        random_latents = th.randn(8, 512).cuda()
        mapped_latents = GENERATOR(random_latents, noise=None, truncation=args.truncation, map_latents=True)
        for latent in mapped_latents:
            ALL_LATENTS.append(latent[None, ...].cpu().numpy())
        batch, _ = GENERATOR(
            styles=mapped_latents,
            noise=None,
            truncation=args.truncation,
            transform_dict_list=[],
            randomize_noise=True,
            input_is_latent=True,
        )
        imgs.append(batch)
    imgs = th.cat(imgs)[:n]
    imgs = F.interpolate(imgs, IMSIZE, mode="bilinear", align_corners=False)
    imgs = (imgs.clamp_(-1, 1) + 1) * 127.5
    imgs = imgs.permute(0, 2, 3, 1)
    imgs_np = imgs.cpu().numpy().astype(np.uint8)
    del imgs
    gc.collect()
    th.cuda.empty_cache()
    return imgs_np


root = tk.Tk()
root.title(name)

imgrid = Scrolling_Area(root, bg="black", width=1680, height=1080)
imgrid.pack(side="left", expand=True, fill="both")

panel = tk.Frame(root, relief="flat", bg="black")
panel.pack(side="right", expand=True, fill="both")


def render_latents(latents):
    imgs = []
    for i in range(latents.shape[0] // 8 + 1):
        if len(latents[8 * i : 8 * (i + 1)]) < 1:
            continue
        batch, _ = GENERATOR(
            styles=latents[8 * i : 8 * (i + 1)].cuda(),
            noise=None,
            truncation=args.truncation,
            transform_dict_list=[],
            randomize_noise=True,
            input_is_latent=True,
        )
        imgs.append(batch)
    imgs = th.cat(imgs)
    imgs = (imgs.clamp_(-1, 1) + 1) / 2
    return imgs


def save():
    intro_latents = np.concatenate(ALL_LATENTS)[INTRO_IDXS]
    torchvision.utils.save_image(
        render_latents(th.from_numpy(intro_latents)),
        f"workspace/{name}_intro_latents.jpg",
        nrow=int(round(math.sqrt(intro_latents.shape[0]) * 4 / 3)),
        padding=0,
        normalize=False,
    )
    np.save(f"workspace/{name}_intro_latents.npy", intro_latents)

    drop_latents = np.concatenate(ALL_LATENTS)[DROP_IDXS]
    torchvision.utils.save_image(
        render_latents(th.from_numpy(drop_latents)),
        f"workspace/{name}_drop_latents.jpg",
        nrow=int(round(math.sqrt(drop_latents.shape[0]) * 4 / 3)),
        padding=0,
        normalize=False,
    )
    np.save(f"workspace/{name}_drop_latents.npy", drop_latents)


tk.Label(panel, text="latents", height=3, bg="black", fg="white").pack(side="top")

but = HoverButton(
    panel,
    text="Save",
    command=save,
    height=3,
    width=8,
    bg="black",
    fg="white",
    activebackground="#333333",
    activeforeground="white",
    relief="flat",
    highlightbackground="#333333",
)
but.pack(side="bottom")

intro = tk.LabelFrame(panel, text="intro", width=240, height=490, bg="black", fg="white", relief="flat")
intro.pack(side="top", fill="both")

drop = tk.LabelFrame(panel, text="drop", width=240, height=490, bg="black", fg="white", relief="flat")
drop.pack(side="bottom", fill="both")

introgrid = Scrolling_Area(intro, width=240, height=490, bg="black")
introgrid.pack(side="top", fill="both")

dropgrid = Scrolling_Area(drop, width=240, height=490, bg="black")
dropgrid.pack(side="bottom", fill="both")

im_num = 0
intro_im_num = 0
drop_im_num = 0


def add_intro(label):
    global intro_im_num
    img_id = int(label.__str__().split(".")[-1])

    INTRO_IDXS.append(img_id)

    img = ImageTk.PhotoImage(image=IMAGES[img_id].resize((46, 46), Image.ANTIALIAS))
    lbl = tk.Label(introgrid.innerframe, image=img, borderwidth=0, highlightthickness=0)
    lbl.image = img  # this line need to prevent gc
    lbl.grid(row=math.floor(intro_im_num / 5), column=intro_im_num % 5)
    lbl.bind("<Button-1>", lambda event, l=lbl: remove_intro(l))

    intro_im_num += 1
    introgrid.update_viewport()


def remove_intro(label):
    remove_idx = list(reversed(introgrid.innerframe.grid_slaves())).index(label)
    label.grid_remove()
    del INTRO_IDXS[remove_idx]
    global intro_im_num
    intro_im_num = 0
    for im in reversed(introgrid.innerframe.grid_slaves()):
        im.grid_remove()
        im.grid(row=math.floor(intro_im_num / 5), column=intro_im_num % 5)
        intro_im_num += 1
    introgrid.update_viewport()


def add_drop(label):
    global drop_im_num
    img_id = int(label.__str__().split(".")[-1])

    DROP_IDXS.append(img_id)

    img = ImageTk.PhotoImage(image=IMAGES[img_id].resize((46, 46), Image.ANTIALIAS))
    lbl = tk.Label(dropgrid.innerframe, image=img, borderwidth=0, highlightthickness=0)
    lbl.image = img  # this line need to prevent gc
    lbl.grid(row=math.floor(drop_im_num / 5), column=drop_im_num % 5)
    lbl.bind("<Button-1>", lambda event, l=lbl: remove_drop(l))

    drop_im_num += 1
    dropgrid.update_viewport()


def remove_drop(label):
    remove_idx = list(reversed(dropgrid.innerframe.grid_slaves())).index(label)
    label.grid_remove()
    del DROP_IDXS[remove_idx]
    global drop_im_num
    drop_im_num = 0
    for im in reversed(dropgrid.innerframe.grid_slaves()):
        im.grid_remove()
        im.grid(row=math.floor(drop_im_num / 5), column=drop_im_num % 5)
        drop_im_num += 1
    dropgrid.update_viewport()


def add_images(n):
    global im_num, IMAGES

    for im_arr in generate_images(n):
        im = Image.fromarray(im_arr)
        IMAGES.append(im)
        img = ImageTk.PhotoImage(image=im)

        label = tk.Label(imgrid.innerframe, image=img, name=str(im_num), borderwidth=0, highlightthickness=0)
        label.image = img  # this line need to prevent gc
        label.grid(row=math.floor(im_num / IMAGES_PER_ROW), column=im_num % IMAGES_PER_ROW)
        label.bind("<Button-1>", lambda event, l=label: add_intro(l))
        label.bind("<Button-3>", lambda event, l=label: add_drop(l))

        im_num += 1

    HoverButton(
        imgrid.innerframe,
        text="More",
        command=lambda n=35: add_images(n),
        height=3,
        width=8,
        bg="black",
        fg="white",
        activebackground="#333333",
        activeforeground="white",
        relief="flat",
        highlightbackground="#333333",
    ).grid(
        row=math.floor(im_num / IMAGES_PER_ROW) + 1,
        column=math.floor(IMAGES_PER_ROW / 2 - 1),
        columnspan=1 if math.floor(IMAGES_PER_ROW / 2 - 1) % 2 == 0 else 2,
    )
    imgrid.update_viewport()


add_images(24)


root.mainloop()
