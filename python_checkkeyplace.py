from statistics import mean
import numpy as np
class CheckKeyPlace:
    def __init__(self, f, key):
        self.key = key
        self.file = f+".txt"
        self.check_key_place()

    def check_key_place(self):
        handle=open(self.file, 'r')
        lines=handle.readlines()
        scores=[]
        for line in lines:
            tokens=line.strip().split()
            linelength=len(tokens)
            for i,token in enumerate(tokens):
                if token==self.key:
                    scores.append(i/(linelength-1) if linelength>=2 else 0)
                    break
        score=mean(scores)
        stderr=np.std(scores, ddof=1) / np.sqrt(len(scores)) if len(scores)>0 else 0
        print("{}\t{}\tscore:{}\tstderr:{}".format(self.file,self.key,score,stderr))



ckp=CheckKeyPlace("output_coco_man", "man")
ckp=CheckKeyPlace("output_coco_toilet", "toilet")
#ckp=CheckKeyPlace("output_coco_man2", "man")
#ckp=CheckKeyPlace("output_coco_toilet2", "toilet")
ckp=CheckKeyPlace("output_enmini_good", "good")
ckp=CheckKeyPlace("output_enmini_bad", "bad")
#ckp=CheckKeyPlace("output_enmini_good2", "good")
#ckp=CheckKeyPlace("output_enmini_bad2", "bad")
ckp=CheckKeyPlace("output_mr_good", "good")
ckp=CheckKeyPlace("output_mr_bad", "bad")
ckp=CheckKeyPlace("output_mr_good2", "good")
ckp=CheckKeyPlace("output_mr_bad2", "bad")
