# from paddleocr import PaddleOCR, draw_ocr
# # Paddleocr目前支持中英文、英文、法语、德语、韩语、日语，可以通过修改lang参数进行切换
# # 参数依次为`ch`, `en`, `french`, `german`, `korean`, `japan`。
# ocr = PaddleOCR(use_angle_cls=True, lang="ch") # need to run only once to download and load model into memory
# img_path = r'C:\Users\021206191\PaddleOCR-develop\pictures\test2.jpg'
# result = ocr.ocr(img_path, cls=True)
# for line in result:
#     print(line)
#
# # 显示结果
# from PIL import Image
# image = Image.open(img_path).convert('RGB')
# boxes = [line[0] for line in result]
# txts = [line[1][0] for line in result]
# scores = [line[1][1] for line in result]
# im_show = draw_ocr(image, boxes, txts, scores, font_path='/path/to/PaddleOCR/doc/simfang.ttf')
# im_show = Image.fromarray(im_show)
# im_show.save('result.jpg')
import string

all_train_word = open('./all_train_word.txt','w+',encoding='UTF-8')
for number in string.digits:
    all_train_word.write(number)
    all_train_word.write('\n')

for word in string.ascii_letters:
    all_train_word.write(word)
    all_train_word.write('\n')

index = 0
for i in range(ord(u'\u4e00'),ord(u'\u9fa5')):
    index +=1
    print(chr(i))
    all_train_word.write(chr(i))
    all_train_word.write('\n')
print(index)
all_train_word.close()