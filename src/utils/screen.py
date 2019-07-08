import cv2 as cv


# From OpenCV example common.py
class Sketcher:
    def __init__(self, windowname, dests, colors, brush):
        self.prev_pt = None
        self.windowname = windowname
        self.dests = dests
        self.colors = colors
        self.brush = brush
        self.dirty = False
        self.show()
        cv.setMouseCallback(self.windowname, self.on_mouse)

    def show(self):
        cv.imshow(self.windowname, self.dests[0])

    def on_mouse(self, event, x, y, flags, param):
        pt = (x, y)
        if event == cv.EVENT_LBUTTONDOWN:
            self.prev_pt = pt
        elif event == cv.EVENT_LBUTTONUP:
            self.prev_pt = None

        if self.prev_pt and flags & cv.EVENT_FLAG_LBUTTON:
            for (dst, color) in zip(self.dests, self.colors):
                cv.line(dst, self.prev_pt, pt, color, thickness=self.brush)
            self.dirty = True
            self.prev_pt = pt
            self.show()
