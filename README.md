Teaching arm path planning project led by Will @ DROP Lab.



def (ptc, image, pixel_class)
pt_i
[x, y, z]
a bunch of image pose (c2w)
K inv(c2w) [x, y, z, 1]
[x_, y_, z_]
[x_/z_, y_/z_] (680, 340) reject if outside range
pt_i (pix_1 (2), pix_2 (1), pix_3 (1).. (20))
for each pixel (cup) [pt_0 cup, pt_28 box, pt_90 table]
    


pt_i (0: 2, 1:1, 2:0, 3:0) => 0

pt_i: class j
4 test images
for each pt_i map back to test images:
    compare class of pti with its corresponding mapped pixel
