"""
解析使用xml.etree.ElementTree 模块，生成使用xml.dom.minidom模块，
ElementTree比dom快，dom生成简单且会自动格式化
"""
import xml.dom.minidom as minidom
import xml.etree.ElementTree as ET
import os
import sys

xmlPath = r'F:\PycharmProjects\VOC2012\Annotations'
jpgPath = r'F:\PycharmProjects\VOC2012\JPEGImages'

def filterXml(fliePath,jpgPath):
    xmlFilePath = os.path.abspath(fliePath)
    # 是否删除xml 和对应的图片
    # 如果没有car标注的话删除xml 和 对应的 jpg
    flag = True

    tree = ET.parse(xmlFilePath)
    # 获得根节点
    root = tree.getroot()
    jpgName = root[1].text
    for child in root:
        if child.tag == 'object' and child[0].text == 'sheep':
            flag = False

    if flag:
        print('不存在 car标注，删除 %s 和 对应的图片%s' % (xmlFilePath, jpgName))
        os.remove(xmlFilePath)
        os.remove(os.path.join(jpgPath, jpgName))

    else:
        for child in root:
            if child.tag == 'object' and child[0].text != 'sheep':
                # 移除节点
                root.remove(child)
        # 保存
        tree.write(xmlFilePath)


def traversalXml(xmlPath, jpgPath):
    for filename in os.listdir(xmlPath):
        xmlFilePath = os.path.join(xmlPath, filename)
        filterXml(xmlFilePath, jpgPath)



traversalXml(xmlPath,jpgPath)