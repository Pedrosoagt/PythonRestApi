import xml.etree.ElementTreee as ET
import datetime

from pymong import MongoClient
from os import rename
from os import listdir
from os.path import isfile, join
from os.path  import getctime

xmlSource='/samba/anonymous_share/spi/SMTB/SPIB'
#xmlSourc='/home/venturus/downloads/test'
xmlTarget='/samba/anonumous_share/spi/SMTB/SPIB/proc'
    
def getBsonHeight(featureResultNode):
    heightNode = featureResultNode.find('Height')
    height = {
                  'Status' : heightNode.attrib['Status'].strip(),
                   'Value' : heightNode.attrib['Value'].strip(),
                  'UpFail' : heightNode.attrib['UpFail'].strip(),
                  'Target' : heightNode['Target'].strip(),
                 'LowFail' : heightNode['LowFail'].strip(),
    }

    return height

def getBsonArea(featureResultNode):
    areaNode = featureResultNode.find('Area')
    area = {
                'Status' : areaNode.attrib['Status'].strip(),
                 'Value' : areaNode.attrib['Value'].strip(),
                'UpFail' : areaNode.attrib['UpFail'].strip(),
                'Target' : areaNode.attrib['Target'].strip(),
               'LowFail' : areaNode.attrib['LowFail'].strip(),
    }

    return area

def getBsonVolume(featureResultNode):
    volumeNode = featureResultNode.find('Volume')
    volume = {
                  'Status' : volumeNode.attrib['Status'].strip(),
                   'Value' : volumeNode.attrib['Value'].strip(),
                  'UpFail' : volumeNode.attrib['UpFail'].strip(),
                  'Target' : volumeNode.attrib['Target'].strip(),
                 'LowFail' : volumeNode.attrib['LowFail'].strip(),
    }

    return volume

def getBsonRegistration(featureResultNode):
    registrationNode = featureResultNode.find('Registration')
    registration = {
                  'Status' : registrationNode.attrib['Status'].strip(),
          'BoundaryStatus' : registrationNode.attrib['BoundaryStatus'].strip(),
                 'LongPct' : registrationNode.attrib['LongPct'].strip(),
            'LongPctLimit' : registrationNode.attrib['LongPctLimit'].strip(),
                'ShortPct' : registrationNode.attrib['ShortPct'].strip(),
           'ShortPctLimit' : registrationNode.attrib['ShortPctLimit'].strip()
    }

    return registration

def getBsonBridging(featureResultNode):
    bridgingNode = featureResultNode.find('Bridging')
    bridging = {
        'Status' : bridgingNode.attrib['Status'].strip(),
         'Value' : bridgingNode.attrib['Value'].strip(),
         'Limit' : bridgingNode.attrib['Limit'].strip()
    }

    return bridging

def getBsonDiagnostics(featureResultNode):
    diagnosticsNode = featureResultNode.find('Diagnostics')
    registrationNode = diagnosticsNode.find('Registration')
    diagnostics = {
        'Registration' : {
             'XOffSet' : registrationNode.attrib['XOffSet'].strip(),
             'YOffSet' : registrationNode.attrib['YOffSet'].strip()
        }
    }

    return diagnostics

def getBsonFeatureResult(featureNode):
    featureResultNode = featureNode.find('FeatureResult')
    featureResult = {
              'Height' : getBsonHeight(featureResultNode),
                'Area' : getBsonArea(featureResultNode),
              'Volume' : getBsonVolume(featureResultNode),
        'Registartion' : getBsonRegistration(featureResultNode),
            'Bridging' : getBsonBridging(featureResultNode),
         'Diagnostics' : getBsonDiagnostics(featureResultNode)
    }

    return featureResult

def getBsonFeatures(locationNode):
    features = []
    for featureNode in locationNode.iter('Feature'):
        feature = {
                     'Name' : locationNode.attrib['Name'].strip(),
                  'LogMode' : locationNode.attrib['LogMode'].strip(),
                   'Status' : locationNode.attrib['Status'].strip(),
                       'Id' : locationNode.attrib['Id'].strip(),
                        'X' : locationNode.attrib['X'].strip(),
                        'Y' : locationNode.attrib['Y'].strip(),
            'FeatureResult' : getBsonFeatureResult(featureNode)
        }
        features.append(feature)

    return features

def getBsonLocations(imageNode):
    locations = []
    for locationNode in imageNode.iter('Location'):
        location = {
                'Name' : locationNode.attrib['Name'].strip(),
                'Part' : locationNode.attrib['Part'].strip(),
             'Package' : locationNode.attrib['Package'].strip(),
                'Code' : locationNode.attrib['Code'].strip(),
            'TestTime' : locationNode.attrib['TestTime'].strip(),
              'Status' : locationNode.attrib['Status'].strip(),
                  'Id' : locationNode.attrib['Id'].strip(),
            'Features' : getBsonFeatures(locationNode)
        }
        locations.append(location)

    return locations

def getBsonImages(panelNode):
    images = []
    for imageNode in panelNode.iter('Image'):
        image = {
                'Name' : imageNode.attrib['Name'].strip(),
                'Code' : imageNode.attrib['Code'].strip(),
            'TestTime' : imageNode.attrib['TestTime'].strip(),
              'Status' : imageNode.attrib['Status'].strip(),
                  'id' : imageNode.attrib['Id'].strip(),
            'Locations': getBsonLocations(imageNode)
        }
        images.append(image)
    
    return images

def getFileModificationTime(fullpath):
    time = getctime(fullpath)
    return datetime.datetime.fromtimestamp(time).isoformat()

cl = MongoClient()
coll = cl["SMTB"]["SPIB"]

fullXmlFiles = [f for f in listdir(xmlSource) if not f.endswith('.r.xml') and f.endswith('.xml') and isfile(join(xmlSource, f))]

for file in fullXmlFiles:
    srcFullPath = join(xmlSource, file)
    targetFullPath = join(xmlTarget, file)

    tree = ET.parse(srcFullPath)

    root = tree.getroot()
    panel = root.find('Panel')

    images = getBsonImages(panel)

    xml = {
        '_id' : file,
        'FileName' : file,
        'FileModification' : getFileModificationTime(srcFullPath),
        'Panel' : {
                     'Name' : panel.attrib['Name'].strip(),
                     'Code' : panel.attrib['Code'].strip(),
               'PanelSizeX' : panel.attrib['PanelSizeX'].strip(),
               'PanelSizeY' : panel.attrib['PanelSizeY'].strip(),
            'PanelRotation' : panel.attrib['PanelRotation'].strip(),
                'StartTime' : panel.attrib['StartTime'].strip(),
                 'TestTime' : panel.attrib['TestTime'].strip(),
                'CycleTime' : panel.attrib['CycleTime'].strip(),
                   'Status' : panel.attrib['Status'].strip(),
                 'SRRFName' : panel.attrib['SRRFName'].strip(),
              'ImagesCount' : len(images),
                   'Images' : images
        }
    }
    coll.insert_one(xml)
    rename(srcFullPath, targetFullPath)
    print(xml['FileName'])

#tree = ET.parse('/home/venturus/donwload/test/SSBENSOL001785900110.#8.5c15609a.xml')

#root = tree.getroot()

#for Image in root.iter('Image')
#   print(Image.attrib)
#   for Location in Image.iter('Location')
#       print(Location.attrib)

doc = { "name" : "bla", "object" : { "name" : "blaa" } }
print(doc['object'])