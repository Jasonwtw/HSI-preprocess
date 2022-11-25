#!/usr/bin/env python
# author: Tingwei Wu
# coding: utf-8


"""
使用方法
    1、下载代码：
        -打开网址：https://github.com/TJIWTR/HSI-preprocess
        -code -> download ZIP
    2、使用代码
        -解压
        -当前文件夹下打开命令行，输入：pip install -r requirements.txt
        -将文件“HSI_preprocess.py”放到本地python环境的包管理文件夹，如：“.\Anaconda\envs\jason\Lib\site-packages\”路径下
        -在自己的工程文件中添加包，如：import HSI_preprocess as HSI
        -使用函数
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import glob
from PIL import Image
import pandas as pd
import shutil
import spectral
import spectral.io.envi as envi
from scipy.signal import savgol_filter


def copyfile(sourcefile, fileclass, destinationfile):
    """
    函数功能：
        从指定目录将指定后缀文件复制到目标文件夹
    参数：
        sourcefile: string, 指定工作目录, 在该目录下遍历
        fileclass: string, 指定文件后缀, 如'.csv'
        destinationfile: string, 目标文件夹
    用法：
        HSI.copyfile('./ori_data/ROI/textile/','.csv','./pre_data/csv/textile/')
    """
    for filenames in os.listdir(sourcefile):
        # 取得文件或文件名的绝对路径
        filepath = os.path.join(sourcefile, filenames)
        # 判断是否为文件夹
        if os.path.isdir(filepath):
            # 如果是文件夹，重新调用该函数
            copyfile(filepath, fileclass, destinationfile)
        # 判断是否为文件
        elif os.path.isfile(filepath):
            # 如果该文件的后缀为用户指定的格式，则把该文件复制到用户指定的目录
            if filepath.endswith(fileclass):
                # filepath_hdr=filepath.replace('.dat','.hdr')
                # 给出提示信息
                print('Copy %s' % filepath + ' To ' + destinationfile)
                if not os.path.exists(destinationfile):
                    os.makedirs(destinationfile)
                # 复制该文件到指定目录
                shutil.copy(filepath, destinationfile)
                # shutil.copy(filepath_hdr,destinationfile)


def readcsv(csvfilepath, newdirpath, start=10, end=210):
    """
    函数功能：
        读取ENVI导出的每一个ROI的csv格式数据，删除无关字符，删除噪声波段，保留10-210波段
    参数：
        csvfilepath: string, csv文件的相对路径或者绝对路径
        newdirpath: string, 处理后数据的保存路径
    用法：
        HSI.readcsv('./ori_data/ROI/leather/2021_10_12_LE_NA/S-LE-1-1.csv', './', 10, 210)
    """
    data_csv = pd.read_csv(csvfilepath, header=None, low_memory=False)
    data_drop = data_csv.drop(index=[0, 1, 2, 3, 4, 5, 6, 7, 8])
    data_iloc = data_drop.iloc[:, (start + 2):(end + 2)]
    if not os.path.exists(newdirpath):
        os.makedirs(newdirpath)
    filename = os.path.basename(csvfilepath)
    savepath = newdirpath + filename
    data_iloc.to_csv(savepath, index=False, header=None, encoding='utf-8-sig')  # 不保存列名和行索引
    return data_iloc


def outlier(csvfilepath, newdirpath):
    """
    函数功能：
        根据3σ原则挑选出异常值并删除，然后覆盖原文件
    参数：
        csvfilepath: string csv文件的相对路径或者绝对路径
        
    用法：
        HSI.outlier('./ori_data/ROI/leather/2021_10_12_LE_NA/S-LE-1-1.csv')
    """
    data_csv = pd.read_csv(csvfilepath, header=None, low_memory=False)
    data_outlier = data_csv.mask((data_csv - data_csv.mean()).abs() > 3 * data_csv.std()).dropna()
    print('原数据尺寸：' + str(data_csv.shape) + '挑选异常值后的尺寸：' + str(data_outlier.shape))
    if not os.path.exists(newdirpath):
        os.makedirs(newdirpath)
    filename = os.path.basename(csvfilepath)
    savepath = newdirpath + filename
    data_outlier_int = data_outlier.astype(int)
    data_outlier_int.to_csv(savepath, index=False, header=None, encoding='utf-8-sig')  # 不保存列名和行索引
    return data_outlier_int


def sgfilter(csvfilepath, newdirpath, window_length=5, polyorder=2):
    """
    函数功能：
        对光谱数据进行S-G滤波变换
    参数：
        csvfilepath: string csv文件的相对路径或者绝对路径
        window_length: int 窗口长度取值为奇数且不能超过len(x)。它越大，则平滑效果越明显；越小，则更贴近原始曲线
        polyorder: int 多项式拟合的阶数。它越小，则平滑效果越明显；越大，则更贴近原始曲线
    
    """
    data_csv = pd.read_csv(csvfilepath, header=None, low_memory=False)
    data_filter_np = savgol_filter(data_csv.to_numpy(), window_length, polyorder, axis=0)
    data_filter_float = pd.DataFrame(data_filter_np)
    data_filter_int = data_filter_float.astype(int)
    if not os.path.exists(newdirpath):
        os.makedirs(newdirpath)
    filename = os.path.basename(csvfilepath)
    savepath = newdirpath + filename
    data_filter_int.to_csv(savepath, index=False, header=None, encoding='utf-8-sig')  # 不保存列名和行索引
    return data_filter_int


def copydata(csvfilepath, newdirpath, fileclass='*.csv', limit=2000):
    """
    函数功能：
        遍历文件夹，找到.csv文件，查找数据量是否满足要求，若不满足则复制两份，满足则保留原数据
    参数：
        csvfilepath: string, 要遍历的文件夹路径
        newdirpath: string, 要保存文件的路径
        fileclass: string, 要查找的文件后缀
        limit: int, 查找的条件
    用法：
        HSI.copydata('./copy/ROI/rubber/', './copy/ROI/rubber/', '*.csv', 2000)
    """
    all_filenames = [i for i in glob.glob(os.path.join(csvfilepath, fileclass))]
    for f in all_filenames:
        data_csv = pd.read_csv(f, header=None, low_memory=False)
        data = []
        if data_csv.shape[0] < limit:
            data.append(data_csv)
            data.append(data_csv)
            data.append(data_csv)
            print(f, '数据少于', limit, '，复制两次')
            if not os.path.exists(newdirpath):
                os.makedirs(newdirpath)
            merged_csv = pd.concat(data, ignore_index=True)
            filename = os.path.basename(f)
            savepath = newdirpath + filename
            merged_csv.to_csv(savepath, index=False, header=None, encoding='utf-8-sig')  # 不保存列名和行索引
            print('保存：', savepath, '成功!')
        else:
            print(f, '数据大于', limit, '，保留原数据')


def preprocess(sourcefile, fileclass, deletepath, outlierpath,
               filterpath, ifreadcsv=True, ifoutlier=True, ifsgfilter=True):
    """
    函数功能：
        遍历指定目录，找到指定后缀文件（主要是.csv)，读取并删掉无用字符，覆盖原文件，再剔除异常值和平滑数据
    参数：
        sourcefile: string, 指定工作目录, 在该目录下遍历
        fileclass: string, 指定文件后缀, 如'.csv'
        outlierpath: string, 指定剔除异常值后的输出路径
        filterpath: string, 指定平滑处理后的输出路径
        ifreadcsv: bool, 是否需要读取csv文件，因为读取csv文件是会裁剪文件的，所以只需要一次，之后注意使用
        ifoutlier,ifsgfilter 同理
    用法：
        HSI.preprocess('D:/data/HSI/ori_data/ROI/background/', '.csv', deletepath='D:/data/HSI/ori_data/ROI/background/',
                      outlierpath='./outlier/background/', 
                      filterpath='./sgfilter/background/', 
                     ifreadcsv=True, ifoutlier=True, ifsgfilter=True)
    """
    # 遍历目录和子目录
    for filenames in os.listdir(sourcefile):
        # 取得文件或文件名的绝对路径
        filepath = os.path.join(sourcefile, filenames)
        # 判断是否为文件夹
        if os.path.isdir(filepath):
            # 如果是文件夹，重新调用该函数
            preprocess(filepath, fileclass, deletepath, outlierpath,
                       filterpath, ifreadcsv=ifreadcsv, ifoutlier=ifoutlier, ifsgfilter=ifsgfilter)
        # 判断是否为文件
        elif os.path.isfile(filepath):
            # 如果该文件的后缀为用户指定的格式，则调用函数处理
            if filepath.endswith(fileclass):
                if ifreadcsv:
                    readcsv(filepath, deletepath)
                    print('读取：%s' % filenames + ' 成功 ')
                    if ifoutlier:
                        csvfilepath = deletepath + filenames
                        outlier(csvfilepath, outlierpath)
                        print('剔除异常值：%s' % filenames + ' 成功 ')
                        if ifsgfilter:
                            sgfilepath = outlierpath + filenames
                            sgfilter(sgfilepath, filterpath)
                            print('平滑：%s' % filenames + ' 成功 ')
                elif ifoutlier:
                    print('无需读取：%s' % filenames)
                    outlier(filepath, outlierpath)
                    print('剔除异常值：%s' % filenames + ' 成功 ')
                    if ifsgfilter:
                        sgfilepath = outlierpath + filenames
                        sgfilter(sgfilepath, filterpath)
                        print('平滑：%s' % filenames + ' 成功 ')
                elif ifsgfilter:
                    print('无需读取+剔除异常值：%s' % filenames)
                    sgfilter(filepath, filterpath)
                    print('平滑：%s' % filenames + ' 成功 ')
    print('预处理成功！')


def plot_spectrums(img_data, class_name):
    """
    函数功能：
        根据数据的平均值和标准差画出光谱图
    参数:
        img_data: 2d-array(numpy) of 3D hyperspectral image
        class_names: string of class name
    返回值:
        mean_spectrums: dict of mean and std spectrum
    """
    # spectrum={}
    step = max(1, img_data.shape[0] // 100)
    fig = plt.figure(figsize=(15, 10))
    for spectrum in img_data[::step, :]:
        plt.plot(spectrum, alpha=0.25)  # alpha 表示透明度
    mean_spectrum = np.mean(img_data, axis=0)
    std_spectrum = np.std(img_data, axis=0)
    lower_spectrum = np.maximum(0, mean_spectrum - std_spectrum)
    higher_spectrum = mean_spectrum + std_spectrum

    plt.fill_between(range(len(mean_spectrum)), lower_spectrum,
                     higher_spectrum, color="#3F5D7D")
    plt.plot(mean_spectrum, alpha=1, color="#FFFFFF", lw=2)
    plt.title(class_name, fontsize=25)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('Wavelength', fontsize=20)
    plt.ylabel('Reflectance*10000', fontsize=20)
    plt.show()
    # spectrum={}
    # return mean_spectrums


def mergecsv(csvfilepath='./', fileclass='*.csv', savepath='./', savecsvname='merged_csv.csv', numofdata=None):
    """
    函数功能：
        合并指定目录下所有.csv的文件，首先判断是否随机选取，如需要则随即采取 numofdata条数据，不需要则选取所有数据，
        最终合并成一个csv文件，并在指定目录下生成csv文件
    参数:
        csvfilepath: string, csv文件的路径
        fileclass: string, 文件的后缀，默认为'*.csv'
        savepath: string, 保存csv文件的路径
        savecsvname: string, 拟保存文件的名字
        numofdata: int, 随机选取每个csv文件的 numofdata行进行合并，默认为 None
    返回值:
        无返回值
    用法：
        HSI.mergecsv('./csv_pre/outlier/paper/', savepath='./data4train_cut/',
                    savecsvname='paper_after_outlier.csv', numofdata=400)
    """
    all_filenames = [i for i in glob.glob(os.path.join(csvfilepath, fileclass))]
    data = []
    for f in all_filenames:
        data_csv = pd.read_csv(f, header=None, low_memory=False)
        if numofdata:
            data_sample = data_csv.sample(numofdata)
            data.append(data_sample)
            print('随机选取csv文件：', os.path.split(f)[1], numofdata, '个数据成功！')
        else:
            data.append(data_csv)
            print('选取csv文件：', os.path.split(f)[1], '全部数据成功！')
    merged_csv = pd.concat(data, ignore_index=True)
    # merged_csv_int = merged_csv.astype(int)
    print('获取数据成功，合并后数据的尺寸为：', merged_csv.shape)
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    merged_csv.to_csv(savepath + savecsvname, index=False, header=None, encoding='utf-8-sig')
    print('成功生成文件：', savepath + savecsvname)
    # 添加了 encoding = ‘utf-8-sig’，以解决导出“非英语”语言时遇到的问题


def insertclass(csvfilepath, classnum, savepath, savecsvname='inserted.csv'):
    """
    函数功能：
        在.csv数据文件的最后一列加上标签阿拉伯数字
    参数:
        csvfilepath: string, csv文件的路径
        classnum: int, 标签的数字
        savepath: string, 保存csv文件的路径
        savecsvname: string, 拟保存文件的名字
    返回值:
        无返回值
    用法：
        HSI.insertclass('./pre_data/csv/sgfilter/wood_2400_21600.csv', 6,
                        './pre_data/csv/sgfilter/insert_class/', 'wood_2400_21600_class.csv')
    """
    data_csv = pd.read_csv(csvfilepath, header=None, low_memory=False)
    data_csv.insert(data_csv.shape[1], '224', classnum)
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    data_csv.to_csv(savepath + savecsvname, index=False, header=None, encoding='utf-8-sig')
    print('%s加入类别标签成功！' % savecsvname)


def clipimg(sourcefile, newdirpath, fileclass='.dat'):
    """
    函数功能：
        遍历指定目录，找到.dat文件，读取并裁剪为 (1070, 640, 224)
    参数：
        sourcefile: string, 指定.dat和.hdr文件所在的目录
        newdirpath: string, 建立文件夹，把裁剪后的数据按原文件名存入
        fileclass: string, 指定文件后缀
    用法：
        HSI.clipimg('./ori_data/HSI_pre/mix/','./pre_data/HSI_cut/mix/')
    """
    # 遍历目录和子目录
    for filenames in os.listdir(sourcefile):
        # 如果该文件的后缀为用户指定的格式，则调用函数处理
        if filenames.endswith(fileclass):
            if not os.path.exists(newdirpath):
                os.makedirs(newdirpath)
            filenames_hdr = filenames.replace('.dat', '.hdr')
            data_dat = envi.open(sourcefile + filenames_hdr, sourcefile + filenames)
            data_clip = data_dat[:1070, :, :]
            savepath = newdirpath + filenames_hdr
            envi.save_image(savepath, data_clip, dtype=np.uint16, force=True)
    print('裁剪图像成功！')


def creatimg(sourcefile, imgclass, fileclass='.img', bands=(50, 115, 190)):
    """
    函数功能：
        遍历指定目录，找到裁剪后的.img文件，读取并创建png图像
    参数：
        sourcefile: string, 指定.dat和.hdr文件所在的目录
        fileclass: string, 指定文件后缀
    用法：
        HSI.creatimg('./pre_data/HSI_cut/mix/','.png')
    """
    for filenames in os.listdir(sourcefile):
        # 如果该文件的后缀为用户指定的格式，则调用函数处理
        if filenames.endswith(fileclass):
            filenames_hdr = filenames.replace(fileclass, '.hdr')
            filenames_img = filenames.replace(fileclass, imgclass)

            data_dat = envi.open(sourcefile + filenames_hdr, sourcefile + filenames)
            rgb = spectral.get_rgb(data_dat, bands)
            rgb /= np.max(rgb)
            rgb_png = np.asarray(255 * rgb, dtype='uint8')
            plt.imsave(sourcefile + filenames_img, rgb_png)
            print('完成保存为png图像！')


def png2jpg(pngdirpath, jpgdirpath):
    """
    函数功能：
        因为labelme对png不友好，需要将png转化为jpg格式
    参数：
        pngdirpath:
        jpgdirpath:
    用法：
        HSI.png2jpg('./labelme/png/', './labelme/jpg/')
    """
    for filenames in os.listdir(pngdirpath):
        # 取得文件或文件名的绝对路径
        filepath = os.path.join(pngdirpath, filenames)
        # 判断是否为文件夹
        if os.path.isdir(filepath):
            # 如果是文件夹，重新调用该函数
            png2jpg(filepath, jpgdirpath)
        # 判断是否为文件
        elif os.path.isfile(filepath):
            # 如果该文件的后缀为用户指定的格式，则把该文件复制到用户指定的目录
            if filepath.endswith('.png'):
                filenames_jpg = filenames.replace('.png', '.jpg')
                # 给出提示信息
                print('Convert %s' % filenames + ' To ' + filenames_jpg)
                if not os.path.exists(jpgdirpath):
                    os.makedirs(jpgdirpath)
                im = Image.open(filepath)
                # 此时返回一个新的image对象，转换图片模式
                image = im.convert('RGB')
                # 调用save()保存
                image.save(jpgdirpath + filenames_jpg)


def create_palette(labels_num):
    """
    函数功能：
        调色板
    Returns:
        palette: dict,
    """
    palette = {0: (0, 0, 0)}  # 将类别为0（ignored_labels）的调色板直接置为(0, 0, 0)，作为背景
    for k, color in enumerate(sns.color_palette("hls", labels_num - 1)):
        # labels_num 是包含了背景的总类别数
        # 构建颜色空间为hls，数目为 labels_num-1 的三元素元组的索引序列（背景已经置为0，所以要减1）
        # color 是三个（0,1）元素组成的元组，代表RGB数值，所以下面要乘以255
        palette[k + 1] = tuple(np.asarray(255 * np.array(color), dtype="uint8"))
    return palette


def convert_to_color(arr_2d, palette=None):
    """
    函数功能：
        将一组标签转换为RGB彩色编码图像
    参数:
        arr_2d: int 2D array of labels
        palette: dict of colors used (label number -> RGB tuple)

    返回值:
        arr_3d: int 2D images of color-encoded labels in RGB format
    """
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)
    if palette is None:
        raise Exception("Unknown color palette")

    for c, i in palette.items():
        m = arr_2d == c
        arr_3d[m] = i

    return arr_3d