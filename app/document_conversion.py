from pymongo import MongoClient

client = MongoClient('localhost')
db = client['DEV']
SPIA_db = db['SPIA']
test = db['SPIA_test']
# SPIA_db_new = db['SPIA_new4']

# SPIB_db = db['SPIB']
# SPIB_db_new = db['SPIB-new4']

print("Number of documents from SPIA: ", db['SPIA'].count_documents({}))
# print("Number of documents from SPIB: ", db['SPIB'].count({}))

print("\n*****\n")

def check_fail(val, up, low=0):
    if float(val) < float(low):
        return 'F', 'Down'
    elif float(val) > float(up):
        return 'F', 'Up'
    else:
        return 'P', 'P'

def check_status(list_result):
    if 'F' in list_result:
        return 'F'
    else:
        return 'P'

def setQuotient(value, target):
    return float(value)/float(target)

def padNormalized(value, low=1, up=1):
    return (float(value) - float(low) )/ ( float(up) - float(low))

def get_feat(result):
    ft = {}
    stats, inf = check_fail(result['Value'], result['UpFail'], result['LowFail'])
    ft['Status'] = stats
    ft['Status_info'] = inf
    ft['Value'] = result['Value']
    ft['UpFail'] = result['UpFail']
    ft['Target'] = result['Target']
    ft['LowFail'] = result['LowFail']
    ft['Quotient'] = setQuotient(result['Value'], result['Target'])
    ft['Normalized'] = padNormalized(result['Value'], result['LowFail'], result['UpFail'])

    return ft

def get_feat_pct(feat_result, pct):
    feat_pct = {}
    feat_pct['Status'] = check_fail(feat_result['{0}Pct'.format(pct)], feat_result['{0}PctLimit'.format(pct)])
    feat_pct['Value'] = feat_result['{0}Pct'.format(pct)]
    feat_pct['Limit'] = feat_result['{0}PctLimit'.format(pct)]
    feat_pct['Quotient'] = setQuotient(feat_result['{0}Pct'.format(pct)], feat_result['{0}PctLimit'.format(pct)])
    feat_pct['Normalized'] = padNormalized(feat_result['{0}Pct'.format(pct)], feat_result['{0}PctLimit'.format(pct)])
    
    return feat_pct

def transform_data(spidb):
    # new_docs_list = []
    count = 0

    for doc in spidb.find():
        new_doc = {}
        # print('Starting new doc!')
        new_doc['_id'] = doc['_id']
        new_doc['Datetime'] = doc['Panel']['StartTime']
        new_doc['TestTime'] = doc['Panel']['TestTime']
        new_doc['Status_XML'] = doc['Panel']['Status']
        new_doc['Name'] = doc['Panel']['Name']
        new_doc['Type'] = doc['Panel']['SRFFName']
        new_doc['Status'] = 'P'

        pads_dict = {}

        img_count = 0

        for img in doc['Panel']['Images']:
            
            img_count += 1
            
            for loc in img['Locations']:
                loc_id = loc['Id']
                loc_part = loc['Part']
                loc_package = loc['Package']
                loc_name = loc['Name']

                for feats in loc['Features']:
                    pad_info = {}
                    pad_info['X'] = feats['X']
                    pad_info['Y'] = feats['Y']
                    pad_info['Id'] = feats['Id']
                    pad_info['Height'] = get_feat(feats['FeatureResult']['Height'])
                    pad_info['Area'] =  get_feat(feats['FeatureResult']['Area'])
                    pad_info['Volume'] = get_feat(feats['FeatureResult']['Volume'])
                    pad_info['ShortPct'] = get_feat_pct(feats['FeatureResult']['Registration'], pct = 'Short')
                    pad_info['LongPct'] = get_feat_pct(feats['FeatureResult']['Registration'], pct = 'Long')
                    pad_info['ComponentId'] = loc_id
                    pad_info['Part'] = loc_part
                    pad_info['Package'] = loc_package
                    pad_info['ComponentName'] = loc_name+"_"+str(img_count)
                    pad_info['ImageCount'] = img_count
                     
                    pad_info['Status'] = check_status([pad_info['Height']['Status'],\
                                                pad_info['Area']['Status'], \
                                                pad_info['Volume']['Status'], \
                                                pad_info['ShortPct']['Status'], \
                                                pad_info['LongPct']['Status']])

                    if pad_info['Status'] is 'F':
                        new_doc['Status'] = 'F'

                    # print(pad_info['Id'])
                    pads_dict[pad_info['Id']] = pad_info

        new_doc['Pad'] = pads_dict

        count += 1
        
        if count % 100 == 0:
            print('New doc created: ', count) 
        
        # new_docs_list.append(new_doc)
        test.insert_one(new_doc)
    return count

print("Begin transformation for SPIA:")
spia_result = transform_data(SPIA_db)
print("Number of insertions: ", spia_result)

print("")

# print("Begin transformation for SPIB:")
# spib_result = transform_data(SPIB_db)

# print("\n*****\n")

# print("Insert SPIA reults to MongoDB:")
# SPIA_db_new.insert_many(spia_result)

# print("")

# print("Insert SPIB reults to MongoDB:")
# SPIB_db_new.insert_many(spib_result)