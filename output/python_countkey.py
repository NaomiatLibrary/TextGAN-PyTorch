from statistics import mean
import numpy as np
class CountKey:
    def __init__(self, f, keys):
        self.keys = keys
        self.file = f+".txt"
        self.count_key()

    def count_key(self):
        handle=open(self.file, 'r')
        lines=handle.readlines()
        for key in self.keys:
            score=0
            for line in lines:
                tokens=line.strip().split()
                for token in tokens:
                    if token == key:
                        score += 1
            print("{} {} {}".format(self.file,key,score))


ck=CountKey("output_coco_nokey", ["man","toilet"])
ck=CountKey("output_coco_man", ["man","toilet"])
ck=CountKey("output_coco_toilet", ["man","toilet"])
ck=CountKey("output_coco_nokey2", ["man","toilet"])
ck=CountKey("output_coco_man2", ["man","toilet"])
ck=CountKey("output_coco_toilet2", ["man","toilet"])
ck=CountKey("output_enmini_nokey", ["good","bad"])
ck=CountKey("output_enmini_good", ["good","bad"])
ck=CountKey("output_enmini_bad", ["good","bad"])
ck=CountKey("output_enmini_nokey2", ["good","bad"])
ck=CountKey("output_enmini_good2", ["good","bad"])
ck=CountKey("output_enmini_bad2", ["good","bad"])
ck=CountKey("output_mr_nokey", ["good","bad"])
ck=CountKey("output_mr_good", ["good","bad"])
ck=CountKey("output_mr_bad", ["good","bad"])
ck=CountKey("output_mr_nokey2", ["good","bad"])
ck=CountKey("output_mr_good2", ["good","bad"])
ck=CountKey("output_mr_bad2", ["good","bad"])
