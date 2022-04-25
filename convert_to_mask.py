import os

src_path = "./Data/Original/F4/"
dst_path = "./Data/Mask/F4/"

if not os.path.exists(dst_path):
   os.makedirs(dst_path)

dirs = os.listdir(src_path)

for item in dirs:
   if item.endswith(".json"):
      if os.path.isfile(src_path + item):
         print("C: " + str(item))
         dst = dst_path + str(item).split('.')[0]
         os.system("labelme_json_to_dataset " + src_path + item + " -o " + dst)
