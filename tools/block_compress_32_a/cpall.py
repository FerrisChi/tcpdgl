import os

FP = "/mnt/data-ssd/yinhbo"
G = ["patents", "lj1", "orkut", "twitter"]
H = [2, 3, 4]
K = [32]
NAME = "res_tc1"

for k in K :
    for h in H :
        for g in G :
            file_path = FP + "/" + g
            graph_path = file_path + "/" + g + "_res.txt"
            cp_cmd = "sudo ./compressor " + graph_path + " " + file_path + "/" + NAME + "_k" + str(k) + "_h" + str(h) + " " + str(k) + " " + str(h)
            print(cp_cmd)
            os.system(cp_cmd)
            print("end")
