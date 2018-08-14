import json

# data1 = {'b': 789, 'c': 456, 'a': 123}
# encode_json = json.dumps(data1)
# print(type(encode_json), encode_json)
#
# decode_json = json.loads(encode_json)
# print(type(decode_json))
# print(decode_json['a'])
# print(decode_json)

my_str = '[{"dev":"ens3","ip":"192.168.232.183","rxkbs":6.361328,"txkbs":8.841797,"mac":"52:54:00:dd:77:c7","netmask":"255.255.255.0","band_width":102400,"link_detected":"yes"}]'
decode_json = json.loads(my_str)[0]
print(type(decode_json))
print(decode_json['rxkbs'])