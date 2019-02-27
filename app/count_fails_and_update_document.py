from pymongo import MongoClient

client = MongoClient('localhost')

db = client['DEV']
collect = db['SPIB_new']

id_list = []
id_list = collect.distinct('_id')

print("Number of different ids: ", len(id_list))

for i in id_list:
    b = collect.find_one({'_id':i})
    pad_info = {}
    pad_info['Total'] = len(b['Pad'].keys())
    count_vol = 0
    count_short_pct = 0
    count_long_pct = 0
    for k in b['Pad'].keys():
        if b['Pad'][k]['Volume']['Status'] == 'F':
            count_vol = count_vol + 1
        if b['Pad'][k]['ShortPct']['Status'] == 'F':
            count_short_pct = count_short_pct + 1
        if b['Pad'][k]['LongPct']['Status'] == 'F':
            count_long_pct = count_long_pct + 1

    info = {}
    info['Volume'] = count_vol
    info['ShortPct'] = count_short_pct
    info['LongPct'] = count_long_pct

    pad_info['Fail'] = info

collect.find_one_and_update({'_id':i}, {'$set': {'Pad_info' : pad_info}})