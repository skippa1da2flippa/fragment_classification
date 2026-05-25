# import cv2
# import matplotlib.pyplot as plt
# import numpy as np
# import inpainter
#
# #---------------------------------------------------------------------
# ## Read image and get its gray scale version to create a binary mask for outpainting.
# def impaint(path):
#     img_rgb = cv2.cvtColor(cv2.imread(path,cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
#     img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
#
#     # resize the image if it is so huge, for faster computation:
#     scale_percent = 50  # percent of original size
#     width = int(img_rgb.shape[1] * scale_percent / 100)
#     height = int(img_rgb.shape[0] * scale_percent / 100)
#     dim = (width, height)
#     img_rgb = cv2.resize(img_rgb, dim, interpolation=cv2.INTER_AREA)
#     img_gray = cv2.resize(img_gray, dim, interpolation=cv2.INTER_AREA)
#
#     #---------------------------------------------------------------------
#     # create a binary mask by thresholding to segment foreground from the background (background is in WHITE (255) color here)
#     ret, mask = cv2.threshold(img_gray, 250, 255, cv2.THRESH_BINARY_INV)
#     # erode the mask from its borders (which improves performance of outpainting)
#     #mask = ~mask
#     #img_rgb[np.where(mask == 0)] = 0
#     kernel = np.ones((5,5))
#     mask = cv2.erode(mask, kernel)
#
#     #P = 10
#     #mask = np.pad(mask, P, 'constant')
#     #img_rgb = cv2.copyMakeBorder(img_rgb, P, P, P, P, cv2.BORDER_CONSTANT, 0)
#
#     ## method of Opencv for inpainting
#     #res1 = cv2.inpaint(
#     #    img_rgb,
#     #    ~mask,
#     #    9,
#     #    cv2.INPAINT_TELEA)
#     #---------------------------------------------------------------------
#     # Inpaint the HIT pixels shown in the binary mask
#     outpainted = inpainter.inpaint(
#         img_rgb.astype(np.uint8),
#         ((~mask) / 255).astype(np.uint8),
#         patch_size=5)
#
#     # to show the outpainted band around the fragment appy dilation to the mask and display it by masking with the outpainted image
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))
#     dilation = cv2.dilate(mask, kernel)
#
#     #---------------------------------------------------------------------
#     # VISUALIZE
#     #plt.imshow(outpainted)
#     #plt.show()
#     #plt.imshow(outpainted & dilation[:, :, None])
#     #plt.show()
#
#     # fix, axes = plt.subplots(1, 2, figsize=(20, 10))
#     # axes[0].imshow(img_rgb)
#     # axes[1].imshow(outpainted & dilation[:, :, None])
#     # plt.show()
#     outpainted = outpainted & dilation[:, :, None]
#     return outpainted
#     #axes[0].set_title(f"num {img}")

import cv2
import matplotlib.pyplot as plt
import numpy as np
import inpainter


def impaint(path):
    # Read with alpha
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    if img.shape[2] == 4:
        # OpenCV reads as BGRA
        bgr = img[:, :, :3]
        alpha = img[:, :, 3]

        img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # Known fragment: alpha > 0
        # Missing/outpainting area: alpha == 0
        known_mask = np.where(alpha > 0, 255, 0).astype(np.uint8)
        inpaint_mask = np.where(alpha == 0, 1, 0).astype(np.uint8)

        # Clean transparent RGB so hidden black does not affect algorithm
        img_rgb[alpha == 0] = [0, 0, 0]

    else:
        # Fallback for non-alpha images
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

        # Old logic: assumes background is white
        _, known_mask = cv2.threshold(
            img_gray,
            250,
            255,
            cv2.THRESH_BINARY_INV
        )

        inpaint_mask = ((~known_mask) / 255).astype(np.uint8)

    # Resize image and masks
    scale_percent = 50
    width = int(img_rgb.shape[1] * scale_percent / 100)
    height = int(img_rgb.shape[0] * scale_percent / 100)
    dim = (width, height)

    img_rgb = cv2.resize(img_rgb, dim, interpolation=cv2.INTER_AREA)
    known_mask = cv2.resize(known_mask, dim, interpolation=cv2.INTER_NEAREST)
    inpaint_mask = cv2.resize(inpaint_mask, dim, interpolation=cv2.INTER_NEAREST)

    # Optional: erode known area slightly, like your old code
    kernel = np.ones((5, 5), np.uint8)
    known_mask = cv2.erode(known_mask, kernel)

    # After erosion, recompute missing/inpaint mask
    inpaint_mask = np.where(known_mask == 0, 1, 0).astype(np.uint8)

    outpainted = inpainter.inpaint(
        img_rgb.astype(np.uint8),
        inpaint_mask,
        patch_size=5
    )

    # Show only band around original fragment
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))
    dilation = cv2.dilate(known_mask, kernel)

    outpainted = outpainted & dilation[:, :, None]

    return outpainted
