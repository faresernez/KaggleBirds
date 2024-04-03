from pathlib import Path
import os

species_dict = {}
size_dict = {}
pathlist = Path(r'C:\Users\fares\OneDrive\Bureau\kaggleBirds\data\train_audio').glob('*')
i = 0
for path in pathlist:
    # because path is object not string
    path_in_str = str(path)
    species = path_in_str[60:]
    species_dict[species] = i
    s = 0
    for ele in os.scandir(path_in_str):
        s += os.path.getsize(ele)
    size_dict[species] = s
    i += 1
sorted_size_dict = sorted(size_dict.items(), key=lambda x:x[1], reverse=True)
converted_dict = dict(sorted_size_dict)

# print(converted_dict)

# print(converted_dict.keys())

sorted_list = list(converted_dict.keys())

batch1 = {}
batch2 = {}
for j in range(11):
    batch2['list_' + str(2*j)] = [sorted_list[12*2*j] ,sorted_list[12*2*j + 2] ,sorted_list[12*2*j + 4] ,sorted_list[12*2*j +6] ,
                                  sorted_list[12*2*j + 8] ,sorted_list[12*2*j + 10] ,sorted_list[12*2*j + 12] ,sorted_list[12*2*j + 14] ,
                                   sorted_list[12*2*j + 16] ,sorted_list[12*2*j + 18] ,sorted_list[12*2*j + 20] ,sorted_list[12*2*j + 22]  ]
    
    batch2['list_' + str(2*j + 1)] =  [sorted_list[12*2*j + 1] ,sorted_list[12*2*j + 3] ,sorted_list[12*j + 5] ,sorted_list[12*2*j +7] ,
                                  sorted_list[12*2*j + 9] ,sorted_list[12*2*j + 11] ,sorted_list[12*2*j + 13] ,sorted_list[12*2*j + 15] ,
                                   sorted_list[12*2*j + 17] ,sorted_list[12*2*j + 19] ,sorted_list[12*2*j + 21] ,sorted_list[12*2*j + 23]  ]
    
    batch1['list_' + str(2*j)] = [sorted_list[12*j] ,sorted_list[12*j + 1] ,sorted_list[12*j + 2] ,sorted_list[12*j +3] ,
                                  sorted_list[12*j + 4] ,sorted_list[12*j + 5] ,sorted_list[12*j + 6] ,sorted_list[12*j + 7] ,
                                   sorted_list[12*j + 8] ,sorted_list[12*j + 9] ,sorted_list[12*j + 10] ,sorted_list[12*j + 11]  ]
    
    batch1['list_' + str(2*j + 1)] = [sorted_list[12*(j+1)] ,sorted_list[12*(j+1) + 1] ,sorted_list[12*(j+1) + 2] ,sorted_list[12*(j+1) +3] ,
                                  sorted_list[12*(j+1) + 4] ,sorted_list[12*(j+1) + 5] ,sorted_list[12*(j+1) + 6] ,sorted_list[12*(j+1) + 7] ,
                                   sorted_list[12*(j+1) + 8] ,sorted_list[12*(j+1) + 9] ,sorted_list[12*(j+1) + 10] ,sorted_list[12*(j+1) + 11]  ]
print(batch1)
print(batch2)
    
