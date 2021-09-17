from PIL import Image
from DealData.FileOperation import FileOperation
import os


class DealImg:
    def __init__(self, save_dir, img_dir):
        self.__fileOp = FileOperation()
        self.__save_dir = save_dir
        self.__img_dir = img_dir

    def run(self):
        person_set = self.__fileOp.get_sub_dirs(self.__img_dir)
        for person in person_set:
            sub_dir = os.path.join(self.__img_dir, person)
            save_sub_dir = os.path.join(self.__save_dir, person)
            self.__fileOp.create_dir(save_sub_dir)
            img_paths = self.__fileOp.scan_all_files(sub_dir)
            id = 0
            for img_path in img_paths:
                extension = self.__fileOp.get_file_extension(img_path)
                if extension!='.png' and extension!='.jpg':
                    print(img_path)
                    continue
                grey_img = self.to_grey(img_path)
                save_img_path = os.path.join(save_sub_dir, str(id) + '.png')
                grey_img.save(save_img_path)
                id += 1
        pass

    def to_grey(self, img_path):
        img = Image.open(img_path)
        box = (200, 100, 1400, 1100)
        tmp_img = img.crop(box)
        grey_img = tmp_img.convert("L")
        grey_img = grey_img.resize((256, 256))
        return grey_img

if __name__ == '__main__':
    save_dir = '../Data/GreyImgs'
    img_dir = '../Data/ROP'
    deal_img = DealImg(save_dir,img_dir)
    deal_img.run()