import torch as th
from torchvision import utils
import torch.utils.cpp_extension
import os

th.ops.load_library("transforms/erode/build/liberode.so")
th.ops.load_library("transforms/dilate/build/libdilate.so")
th.ops.load_library("transforms/scale/build/libscale.so")
th.ops.load_library("transforms/rotate/build/librotate.so")
th.ops.load_library("transforms/translate/build/libtranslate.so")


class Erode(th.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, batch, params, indicies):
        if not isinstance(params[0], int) or params[0] < 0:
            print("Erosion parameter must be a positive integer")
            # raise ValueError
        new_xs = []
        for x in batch:
            x_array = list(th.split(x, 1, 1))
            for i, dim in enumerate(x_array):
                if indicies == "all" or i in indicies:
                    d_ = th.squeeze(dim)
                    tf = th.ops.my_ops.erode(d_, params[0])
                    tf = th.unsqueeze(th.unsqueeze(tf, 0), 0)
                    x_array[i] = tf
            new_xs.append(th.cat(x_array, 1))
        return th.cat(new_xs, 0).reshape(batch.shape)


class Dilate(th.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, batch, params, indicies):
        if not isinstance(params[0], int) or params[0] < 0:
            print("Dilation parameter must be a positive integer")
            # raise ValueError
        new_xs = []
        for x in batch:
            x_array = list(th.split(x, 1, 1))
            for i, dim in enumerate(x_array):
                if indicies == "all" or i in indicies:
                    d_ = th.squeeze(dim)
                    tf = th.ops.my_ops.dilate(d_, params[0])
                    tf = th.unsqueeze(th.unsqueeze(tf, 0), 0)
                    x_array[i] = tf
            new_xs.append(th.cat(x_array, 1))
        return th.cat(new_xs, 0).reshape(batch.shape)


class Translate(th.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, batch, params, indicies):
        if (
            not isinstance(params[0], float)
            or not isinstance(params[1], float)
            or params[0] < -1
            or params[0] > 1
            or params[1] < -1
            or params[1] > 1
        ):
            print("Translation must have two parameters, which should be floats between -1 and 1.")
            # raise ValueError
        # print("batch", batch.shape)
        print(batch.min(), batch.mean(), batch.max(), batch.shape)
        new_xs = []
        for x in batch:
            x_array = list(th.split(x, 1, 1))
            for i, dim in enumerate(x_array):
                if indicies == "all" or i in indicies:
                    d_ = th.squeeze(dim)
                    tf = th.ops.my_ops.translate(d_, params[0], params[1])
                    tf = th.unsqueeze(th.unsqueeze(tf, 0), 0)
                    x_array[i] = tf
            new_xs.append(th.cat(x_array, 1))
        # print("batch", th.cat(new_xs, 0).reshape(batch.shape).shape)
        output = th.cat(new_xs, 0).reshape(batch.shape)
        print(output.min(), output.mean(), output.max(), output.shape)
        return output


class Scale(th.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, batch, params, indicies):
        if not isinstance(params[0], float):
            print("Scale parameter should be a float.")
        new_xs = []
        for x in batch:  # raise ValueError
            x_array = list(th.split(x, 1, 1))
            for i, dim in enumerate(x_array):
                if indicies == "all" or i in indicies:
                    d_ = th.squeeze(dim)
                    tf = th.ops.my_ops.scale(d_, params[0])
                    tf = th.unsqueeze(th.unsqueeze(tf, 0), 0)
                    x_array[i] = tf
            new_xs.append(th.cat(x_array, 1))
        return th.cat(new_xs, 0).reshape(batch.shape)


class Rotate(th.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, batch, params, indicies):
        if not isinstance(params[0], float) or params[0] < 0 or params[0] > 360:
            print("Rotation parameter should be a float between 0 and 360 degrees.")
            # raise ValueError
        print("batch", batch.shape)
        new_xs = []
        for x in batch:
            x_array = list(th.split(x, 1, 1))
            for i, dim in enumerate(x_array):
                if indicies == "all" or i in indicies:
                    d_ = th.squeeze(dim)
                    tf = th.ops.my_ops.rotate(d_, params[0])
                    tf = th.unsqueeze(th.unsqueeze(tf, 0), 0)
                    x_array[i] = tf
            new_xs.append(th.cat(x_array, 1))
        print("batch", th.cat(new_xs, 0).reshape(batch.shape).shape)
        return th.cat(new_xs, 0).reshape(batch.shape)


class FlipHorizontal(th.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, batch, params, indicies):
        new_xs = []
        for x in batch:
            x_array = list(th.split(x, 1, 1))
            for i, dim in enumerate(x_array):
                if indicies == "all" or i in indicies:
                    d_ = th.squeeze(dim)
                    tf = d_.flip([1])
                    tf = th.unsqueeze(th.unsqueeze(tf, 0), 0)
                    x_array[i] = tf
            new_xs.append(th.cat(x_array, 1))
        return th.cat(new_xs, 0).reshape(batch.shape)


class FlipVertical(th.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, batch, params, indicies):
        new_xs = []
        for x in batch:
            x_array = list(th.split(x, 1, 1))
            for i, dim in enumerate(x_array):
                if indicies == "all" or i in indicies:
                    d_ = th.squeeze(dim)
                    tf = d_.flip([0])
                    tf = th.unsqueeze(th.unsqueeze(tf, 0), 0)
                    x_array[i] = tf
            new_xs.append(th.cat(x_array, 1))
        return th.cat(new_xs, 0).reshape(batch.shape)


class Invert(th.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, batch, params, indicies):
        new_xs = []
        for x in batch:
            x_array = list(th.split(x, 1, 1))
            for i, dim in enumerate(x_array):
                if indicies == "all" or i in indicies:
                    d_ = th.squeeze(dim)
                    ones = th.ones(d_.size(), dtype=d_.dtype, layout=d_.layout, device=d_.device)
                    tf = ones - d_
                    tf = th.unsqueeze(th.unsqueeze(tf, 0), 0)
                    x_array[i] = tf
            new_xs.append(th.cat(x_array, 1))
        return th.cat(new_xs, 0).reshape(batch.shape)


class BinaryThreshold(th.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, batch, params, indicies):
        if not isinstance(params[0], float) or params[0] < -1 or params[0] > 1:
            print("Binary threshold parameter should be a float between -1 and 1.")
            # raise ValueError
        new_xs = []
        for x in batch:
            x_array = list(th.split(x, 1, 1))
            for i, dim in enumerate(x_array):
                if indicies == "all" or i in indicies:
                    d_ = th.squeeze(dim)
                    t = th.autograd.Variable(th.Tensor([params[0]]))
                    t = t.to(d_.device)
                    tf = (d_ > t).float() * 1
                    tf = th.unsqueeze(th.unsqueeze(tf, 0), 0)
                    x_array[i] = tf
            new_xs.append(th.cat(x_array, 1))
        return th.cat(new_xs, 0).reshape(batch.shape)


class ScalarMultiply(th.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, batch, params, indicies):
        if not isinstance(params[0], float):
            print("Scalar multiply parameter should be a float")
            # raise ValueError
        new_xs = []
        for x in batch:
            x_array = list(th.split(x, 1, 1))
            for i, dim in enumerate(x_array):
                if indicies == "all" or i in indicies:
                    d_ = th.squeeze(dim)
                    tf = d_ * params[0]
                    tf = th.unsqueeze(th.unsqueeze(tf, 0), 0)
                    x_array[i] = tf
            new_xs.append(th.cat(x_array, 1))
        return th.cat(new_xs, 0).reshape(batch.shape)


class Ablate(th.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, batch, params, indicies):
        new_xs = []
        for x in batch:
            x_array = list(th.split(x, 1, 1))
            for i, dim in enumerate(x_array):
                if indicies == "all" or i in indicies:
                    d_ = th.squeeze(dim)
                    tf = d_ * 0
                    tf = th.unsqueeze(th.unsqueeze(tf, 0), 0)
                    x_array[i] = tf
            new_xs.append(th.cat(x_array, 1))
        return th.cat(new_xs, 0).reshape(batch.shape)


class DoubleWidth(th.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, params, indicies):
        _, _, _, w = x.shape
        x = th.cat(
            [
                x[:, :, :, : w // 2 + 1][:, :, :, list(range(w // 2, 0, -1))],  # first half mirrored
                x,  #                                                           middle section
                x[:, :, :, w // 2 :][:, :, :, list(range(w // 2, 0, -1))],  # second half mirrored
            ],
            axis=3,
        )
        return x


class ManipulationLayer(th.nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

        # layers
        self.erode = Erode()
        self.dilate = Dilate()
        self.translate = Translate()
        self.scale = Scale()
        self.rotate = Rotate()
        self.flip_h = FlipHorizontal()
        self.flip_v = FlipVertical()
        self.invert = Invert()
        self.binary_thresh = BinaryThreshold()
        self.scalar_multiply = ScalarMultiply()
        self.ablate = Ablate()
        self.double_width = DoubleWidth()

        self.layer_options = {
            "erode": self.erode,
            "dilate": self.dilate,
            "translate": self.translate,
            "scale": self.scale,
            "rotate": self.rotate,
            "flip-h": self.flip_h,
            "flip-v": self.flip_v,
            "invert": self.invert,
            "binary-thresh": self.binary_thresh,
            "scalar-multiply": self.scalar_multiply,
            "ablate": self.ablate,
            "double-width": self.double_width,
        }

    def save_activations(self, input, index):
        path = "activations/" + str(self.layer) + "/" + str(index) + "/"
        if not os.path.exists(path):
            os.makedirs(path)

        x_array = list(th.split(input, 1, 1))
        for i, activation in enumerate(x_array):
            utils.save_image(
                th.squeeze(activation), path + str(i).zfill(3) + ".png", nrow=1, normalize=True, range=(-1, 1)
            )

    def forward(self, input, tranforms_dict_list):
        out = input
        for transform_dict in tranforms_dict_list:
            # if transform_dict["layer"] == -1:
            #     self.save_activations(input, transform_dict["index"])
            if transform_dict["layer"] == self.layer:
                # print(f"applying {transform_dict['transform']}({transform_dict['params']}) on layer {self.layer}")
                out = self.layer_options[transform_dict["transform"]](
                    out, transform_dict["params"], transform_dict["indicies"]
                )
        return out

