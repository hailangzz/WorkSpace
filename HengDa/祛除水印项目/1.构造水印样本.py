# coding:utf-8

from PIL import Image, ImageDraw, ImageFont
import random


def add_text_to_image(image, text):
    font = ImageFont.truetype('C:\Windows\Fonts\STXINGKA.TTF', 36)

    #     # 添加背景
    #     new_img = Image.new('RGBA', (image.size[0] * 3, image.size[1] * 3), (0, 0, 0, 0))  #创建三倍于原图大小的图像
    #     new_img.paste(image, image.size) # 将图像叠加到另一张图像上
    new_img = image
    #     new_img.show()

    # 添加水印
    font_len = len(text)
    rgba_image = new_img.convert('RGBA')
    text_overlay = Image.new('RGBA', rgba_image.size, (255, 255, 255, 0))
    image_draw = ImageDraw.Draw(text_overlay)  # 创建一个可以在给定图像上绘图的对象···

    #     rgba_image.show()
    # 随机贴上水印····

    cut_point = {'x_cut': 0, 'y_cut': 0}
    water_flag_numbers = 10000
    for numbers in range(water_flag_numbers):
        text_overlay = Image.new('RGBA', rgba_image.size, (255, 255, 255, 0))
        image_draw = ImageDraw.Draw(text_overlay)  # 创建一个可以在给定图像上绘图的对象···
        x_point = random.randint(0, new_img.size[0] - 40 * font_len)
        y_point = random.randint(0, new_img.size[1] - 40)

        cut_point['x_cut'] = x_point
        cut_point['y_cut'] = y_point
        image_draw.text((x_point, y_point), text, font=font, fill=(0, 0, 0, 50))  # 在图像上的特定位置绘制图形···
        text_overlay = text_overlay.rotate(0)  # 旋转绘制的图像
        image_with_text = Image.alpha_composite(rgba_image, text_overlay)  # 按照一定透明度贴合图片···


        image_with_text = image_with_text.crop((cut_point['x_cut'], cut_point['y_cut'],
                                                cut_point['x_cut'] + 36 * font_len, cut_point['y_cut'] + 36))  # 切割需要的图片

        image_with_text.save(u'F:\\water_picture_program\\yes_water_picture\\' + str(numbers) + '_water_picture.png')

        test_sample = rgba_image.crop((cut_point['x_cut'], cut_point['y_cut'], cut_point['x_cut'] + 36 * font_len,
                                       cut_point['y_cut'] + 36))  # 切割需要的图片
        test_sample.save(u'F:\\water_picture_program\\not_water_picture\\' + str(numbers) + '_water_picture.png')

        #         return image_with_text


if __name__ == '__main__':
    img = Image.open("F:\行程卡.jpg")
    im_after = add_text_to_image(img, u'石家庄')
#     im_after.save(u'F:\水印.png')