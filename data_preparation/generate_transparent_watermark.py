import os
import os.path as osp
from PIL import Image, ImageFont, ImageDraw


def addTextWatermark(image_path, font_path, output_path, text, show=False):
    """Add text watermark to the given image

    Arguments:
        image_path {str} -- path to image
        font_path {str} -- path to the font used in watermark
        output_path {str} -- path to the output folder
        text {str} -- text on the watermark

    Keyword Arguments:
        show {bool} -- show the origin image, generated watermark and the image with watermark (default: {False})
    """
    image_file_name = osp.basename(image_path)
    image_file_name_without_ext = osp.splitext(image_file_name)[0]

    mask_output = osp.join(output_path, "mask")
    masked_output = osp.join(output_path, "masked")
    if not osp.exists(mask_output):
        os.makedirs(mask_output)
    if not osp.exists(masked_output):
        os.makedirs(masked_output)

    image = Image.open(image_path).convert("RGBA")
    if show:
        image.show()

    image_w, image_h = image.size
    margin_left, margin_bottom = 10, 10

    font = ImageFont.truetype(font_path, 12)
    text_w, text_h = font.getsize(text)

    text_mask = Image.new("RGB", image.size, (255, 255, 255))
    draw_text_mask = ImageDraw.Draw(text_mask)
    draw_text_mask.text((image_w - margin_left - text_w, image_h - margin_bottom - text_h),
                        text, fill=(0, 0, 0), font=font)
    text_mask.save(
        osp.join(mask_output, "{}_text_mask.png".format(image_file_name_without_ext)))
    if show:
        text_mask.show()

    text_rect_mask = Image.new("RGB", image.size, (255, 255, 255))
    draw_text_rect_mask = ImageDraw.Draw(text_rect_mask)
    draw_text_rect_mask.rectangle([(image_w - margin_left - text_w, image_h - margin_bottom - text_h),
                                   (image_w - margin_left, image_h - margin_bottom)],
                                  fill=(0, 0, 0), outline=(0, 0, 0))
    text_rect_mask.save(
        osp.join(mask_output, "{}_rect_mask.png".format(image_file_name_without_ext)))

    text_watermark = Image.new("RGBA", image.size, (255, 255, 255, 0))
    draw_text_watermark = ImageDraw.Draw(text_watermark)
    draw_text_watermark.text((image_w - margin_left - text_w, image_h - margin_bottom - text_h),
                             text, fill=(255, 255, 255, 192), font=font)
    out = Image.alpha_composite(image, text_watermark)
    if show:
        out.show()
    out.save(
        osp.join(masked_output, "{}_transparent_masked.png".format(image_file_name_without_ext)))
    out.convert("RGB").save(
        osp.join(masked_output, "{}_masked.png".format(image_file_name_without_ext)))


if __name__ == "__main__":
    image_path = r"data_preparation\download.jpg"
    font_path = r"data_preparation\msyh.ttc"
    output_path = r"data_preparation\result"
    text = r"知乎 @张宇星"
    show = False
    addTextWatermark(image_path, font_path, output_path, text, show)
