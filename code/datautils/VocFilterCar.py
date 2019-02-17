import xml.etree.ElementTree as ET
import os

xmlPath = r'F:\PycharmProjects\VOC2012\Annotations'
jpgPath = r'F:\PycharmProjects\VOC2012\JPEGImages'

def filterXml(fliePath,jpgPath):
    try:
        xmlFilePath = os.path.abspath(fliePath)
        # 是否删除xml 和对应的图片
        # 如果没有car标注的话删除xml 和 对应的 jpg
        flag = True

        tree = ET.parse(xmlFilePath)
        # 获得根节点
        root = tree.getroot()
        for child in root:
            if child.tag == 'filename':
                jpgName = child.text
            if child.tag == 'object' and child[0].text == 'car':
                flag = False

        if flag:
            print('不存在 car标注，删除 %s 和 对应的图片%s' % (xmlFilePath, jpgName))

            os.remove(xmlFilePath)
            os.remove(os.path.join(jpgPath, jpgName))

        else:
            for child in root:
                if child.tag == 'object' and child[0].text != 'car':
                    # 移除节点
                    root.remove(child)
                    # 保存
                    tree.write(xmlFilePath)
    except Exception as e:
        print(e)

def traversalXml(xmlPath, jpgPath):
    for filename in os.listdir(xmlPath):
        xmlFilePath = os.path.join(xmlPath, filename)
        filterXml(xmlFilePath, jpgPath)



traversalXml(xmlPath,jpgPath)