# coding:utf-8

from PIL import Image, ImageDraw, ImageFont
import random
import copy
import os

def add_text_to_image(image_path, text):
    file_path = r'F:\工作文档\祛除水印项目\水印检测标注样本\\'
    image_name = image_path.split('\\')[-1].split('.')[0]+'.png'
    print(image_name)

    image = Image.open(image_path)
    font = ImageFont.truetype('C:\Windows\Fonts\STXINGKA.TTF', 36)

    #     # 添加背景
    #     new_img = Image.new('RGBA', (image.size[0] * 3, image.size[1] * 3), (0, 0, 0, 0))  #创建三倍于原图大小的图像
    #     new_img.paste(image, image.size) # 将图像叠加到另一张图像上
    new_img = image
    #     new_img.show()

    # 添加水印
    font_len = len(text)
    rgba_image = new_img.convert('RGBA')
    # text_overlay = Image.new('RGBA', rgba_image.size, (255, 255, 255, 0))
    # image_draw = ImageDraw.Draw(text_overlay)  # 创建一个可以在给定图像上绘图的对象···

    #     rgba_image.show()
    # 随机贴上水印····

    cut_point_list = []
    background_picture_numbers = 1
    water_flag_numbers = 10
    det_flag_info_total_list=[] # 水印检测标记框的记录信息列表
    det_flag_info_singel_dict = {'transcription':'','points':[]}


    cut_point_single = {'x0_cut': 0,'x1_cut': 0, 'y0_cut': 0, 'y1_cut': 0}
    image_with_text = rgba_image
    for numbers in range(water_flag_numbers): #每张图贴合5张水印···
        text_overlay = Image.new('RGBA', rgba_image.size, (255, 255, 255, 0))
        image_draw = ImageDraw.Draw(text_overlay)  # 创建一个可以在给定图像上绘图的对象···
        x_point = random.randint(0, new_img.size[0] - 40 * font_len)
        y_point = random.randint(0, new_img.size[1] - 40)

        cut_point_single['x0_cut'] = x_point
        cut_point_single['y0_cut'] = y_point
        cut_point_single['x1_cut'] = x_point + 36 * font_len
        cut_point_single['y1_cut'] = y_point + 36
        cut_point_list.append(copy.deepcopy(cut_point_single))

        # 写入水印的位置及文本信息···
        det_flag_info_singel_dict['transcription'] = text
        water_mark_location = [
                            [cut_point_single['x0_cut'], cut_point_single['y0_cut']],
                            [cut_point_single['x1_cut'], cut_point_single['y0_cut']],
                            [cut_point_single['x1_cut'], cut_point_single['y1_cut']],
                            [cut_point_single['x0_cut'], cut_point_single['y1_cut']],
                               ]
        det_flag_info_singel_dict['points']=water_mark_location
        det_flag_info_total_list.append(copy.deepcopy(det_flag_info_singel_dict))


        image_draw.text((x_point, y_point), text, font=font, fill=(0, 0, 0, 50))  # 在图像上的特定位置绘制图形···
        text_overlay = text_overlay.rotate(0)  # 旋转绘制的图像
        image_with_text = Image.alpha_composite(image_with_text, text_overlay)  # 按照一定透明度贴合图片···

        # 切割水印子图，用于祛除水印模型训练··
        image_with_text_part = image_with_text.crop((cut_point_single['x0_cut'], cut_point_single['y0_cut'],
                                                cut_point_single['x0_cut'] + 36 * font_len, cut_point_single['y0_cut'] + 36))  # 切割需要的图片

        image_with_text_part.save(u'F:\\water_picture_program_model\\yes_water_picture\\' + image_name + str(numbers) + '_water_picture.png')

        test_sample = rgba_image.crop((cut_point_single['x0_cut'], cut_point_single['y0_cut'], cut_point_single['x0_cut'] + 36 * font_len,
                                       cut_point_single['y0_cut'] + 36))  # 切割需要的图片
        test_sample.save(u'F:\\water_picture_program_model\\not_water_picture\\' + image_name +  str(numbers) + '_water_picture.png')


    write_det_flag_mark_file(image_name,det_flag_info_total_list)
    image_with_text.save(file_path+'det_mark_'+image_name)


def write_det_flag_mark_file(pictrue_name,cut_point_list,file_path = r'F:\工作文档\祛除水印项目\水印检测标注样本\det_train_water_flag_mark_lable.txt'):#写入水印检测标注文件···
    det_flag_file_cur = open(r'F:\工作文档\祛除水印项目\水印检测标注样本\det_train_water_flag_mark_lable.txt', 'a+',encoding='utf-8')

    det_flag_file_cur.write('det_mark_'+pictrue_name+'\t')
    det_flag_file_cur.write(str(cut_point_list))
    det_flag_file_cur.write('\n')
    det_flag_file_cur.close()

def get_backgroud_picture_path(origin_path = r'C:\Users\021206191\PaddleOCR-develop\train_data\icdar2015\text_localization\icdar_c4_train_imgs'):
    background_picture_name = os.listdir(origin_path)
    background_picture_list = [os.path.join(origin_path,single_picture_name) for single_picture_name in background_picture_name]
    return background_picture_list

if __name__ == '__main__':

    background_picture_list = get_backgroud_picture_path()
    for index in range(1000):
        image_path=random.sample(background_picture_list,1)[0]
        im_after = add_text_to_image(image_path, u'石家庄')
