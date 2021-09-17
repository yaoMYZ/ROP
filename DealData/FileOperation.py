# -*- coding: utf-8 -*-
'''
Created on Sep 11, 2017
Modify on Oct 28, 2017

@author: yao
'''
import csv
import os
import sys
import xlrd
import zipfile
import tarfile


class FileOperation:
    def __init__(self):
        self.__txtFileOp = None
        self.__scvFileOp = None

    # 写入一行或多行数据(模式为直接写入，若已存在，删除已有信息，重新写入)
    def write_csv(self, data, output_file_path):
        csvfile = open(output_file_path, 'w', newline='')
        writer = csv.writer(csvfile)

        self.__write_csv_data(writer, data)
        csvfile.close()

    # 写入一行或多行数据(模式为从尾部添加)
    def write_csv_by_append(self, data, output_file_path):
        csvfile = open(output_file_path, 'a', newline='')
        writer = csv.writer(csvfile)
        self.__write_csv_data(writer, data)

        csvfile.close()

    def __write_csv_data(self, writer, data):
        if type(data[0])!=list:
            writer.writerow(data)
        else:
            writer.writerows(data)

    # 获取dirname目录下所有文件名(不包括其子目录)
    def scan_current_path_files(self, dirname):
        all_filename = []
        # 返回一个列表，其中包含在文件夹和文件的名称
        filenames = os.listdir(dirname)
        for filename in filenames:
            if os.path.isfile(dirname + '/' + filename):
                if filename[0] != '.':  # 排除隐藏文件
                    all_filename.append(filename)
        return all_filename


    def get_sub_dirs(self,parent_dir):
        sub_dirs = []
        if (os.path.exists(parent_dir)):
            # 获取该目录下的所有文件或文件夹目录
            files = os.listdir(parent_dir)
            for file in files:
                m = os.path.join(parent_dir, file)
                # 判断该路径下是否是文件夹
                if (os.path.isdir(m)):
                    h = os.path.split(m)
                    sub_dirs.append(h[1])
        return sub_dirs


    # 获取dirname目录及其子目录下所有文件的路径
    def scan_all_files(self, dirname):
        all_filepath = []
        # 三个参数：分别返回1.父目录 2.所有文件夹名字（不含路径） 3.所有文件名字
        parent_dir_index = 0
        filename_index = 2
        for fileContents in os.walk(dirname):
            for filename in fileContents[filename_index]:
                if len(filename) != 0 and filename[0] != '.':  # 排除空文件和隐藏文件
                    all_filepath.append(fileContents[parent_dir_index] + '/' + filename)  # 输出文件信息

        return all_filepath


    def read_table(self,input_file_path):
        extension=self.get_file_extension(input_file_path)
        if extension=='.csv':
            return self.read_csv(input_file_path)
        if extension=='.xls' or extension=='.xlsx':
            return self.read_excel(input_file_path)
        return False


    def read_csv(self, input_file_path):
        with open(input_file_path,'r',encoding="utf-8") as csvfile:
            reader = csv.reader(csvfile)
            All = [row for row in reader]

        return All


    def get_sheet_names(self,input_file_path):
        tables = xlrd.open_workbook(input_file_path)
        return  tables.sheet_names()

    def read_excel(self,input_file_path,sheet_name=None):
        tables = xlrd.open_workbook(input_file_path)
        if sheet_name==None:
            table = tables.sheet_by_index(0)  # 通过索引顺序获取
        else:
            table=tables.sheet_by_name(sheet_name)
        nrows=table.nrows
        datas=[]
        for i in range(nrows):
            datas.append(table.row_values(i))
        return datas


    def start_read_txt_by_line(self, input_file_path):
        self.__txtFileOp = open(input_file_path)

    def end_read_txt_by_line(self):
        self.__txtFileOp.close()
        self.__txtFileOp = None

    def get_txt_line(self, exclude_line_break=True):
        line = self.__txtFileOp.readline()
        if exclude_line_break:
            line = line.rstrip('\n')  # 去除换行符
        return line



    def read_txt(self, input_file_path, exclude_line_break):
        txtfile = open(input_file_path, 'r')
        content = []
        for line in txtfile:
            if exclude_line_break:
                line = line.rstrip('\n')  # 去除换行符
            content.append(line)
        txtfile.close()
        return content


    def write_txt(self, data, output_file_path):
        txtfile = open(output_file_path, 'w')
        self.__write_txt_data(txtfile, data)

        txtfile.close()

    def write_txt_by_append(self, data, output_file_path):
        txtfile = open(output_file_path, 'a')
        self.__write_txt_data(txtfile, data)

        txtfile.close()


    def __write_txt_data(self, txtfile, data):
        if type(data) == str:
            txtfile.write(data)
        else:
            txtfile.writelines(data)




    def decompress_file(self,compress_file_path,extract_dir):
        extension = self.get_file_extension(compress_file_path)
        if extension == '.zip':
            return self.decompress_zip(compress_file_path,extract_dir)
        if extension == '.gz':
            return self.decompress_targz(compress_file_path,extract_dir)
        return False




    def decompress_zip(self,compress_file_path,extract_dir):
        f = zipfile.ZipFile(compress_file_path, 'r')
        f.extractall(extract_dir)
        f.close()
        # for file in f.namelist():
        #     f.extract(file,extract_dir )
        return True




    def decompress_targz(self,compress_file_path,extract_dir):
        tarHandle = tarfile.open(compress_file_path, "r:gz")
        tarHandle.extractall(extract_dir)
        tarHandle.close()
        return True


    def get_file_extension(self,path):
        return os.path.splitext(path)[1]



    def get_parent_dir(self,file_path):
        return os.path.abspath(os.path.dirname(file_path) + os.path.sep + ".")

    def delete_file(self,file_path):
        os.remove(file_path)

    def get_file_name(self,file_path,remove_uffix=True):
        filename= os.path.basename(file_path)
        if remove_uffix:
            return os.path.splitext(filename)[0]
        else:
            return filename

    def remove_uffix(self,file_path):
        return os.path.splitext(file_path)[0]

    def create_dir(self,dir):
        if not os.path.exists(dir):
            os.makedirs(dir)










