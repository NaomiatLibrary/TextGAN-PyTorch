from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("ggplot")     # グラフのデザインを指定する
sns.set_palette('Set2')     # グラフの色を指定する
import warnings
warnings.filterwarnings('ignore') # 警告メッセージを出ないようにしている

hakohige_labels=[]
hakohige_lawdata=[]
hakohige_data=[]
class CheckKeyPlace:
    def __init__(self, f, key):
        self.key = key
        self.file = f+".txt"
        self.check_key_place()

    def check_key_place(self):
        global hakohige_labels
        global hakohige_lawdata
        global hakohige_data
        handle=open(self.file, 'r')
        lines=handle.readlines()
        scores=[]
        lawscores=[]
        for line in lines:
            tokens=line.strip().split()
            linelength=len(tokens)
            for i,token in enumerate(tokens):
                if token==self.key:
                    scores.append(i/(linelength-1) if linelength>=2 else 0)
                    lawscores.append(i)
                    break
        score=mean(scores)
        hakohige_labels.append(self.file[7:-4])
        hakohige_lawdata.append(lawscores)
        hakohige_data.append(scores)
        stderr=np.std(scores, ddof=1) / np.sqrt(len(scores)) if len(scores)>0 else 0
        print("{} {} score:{} stderr:{}".format(self.file,self.key,score,stderr))


ckp=CheckKeyPlace("output_mr_good", "good")
ckp=CheckKeyPlace("output_mr_good2", "good")
ckp=CheckKeyPlace("output_mr_bad", "bad")
ckp=CheckKeyPlace("output_mr_bad2", "bad")
ckp=CheckKeyPlace("output_enmini_good", "good")
ckp=CheckKeyPlace("output_enmini_good2", "good")
ckp=CheckKeyPlace("output_enmini_bad", "bad")
ckp=CheckKeyPlace("output_enmini_bad2", "bad")
ckp=CheckKeyPlace("output_coco_man", "man")
ckp=CheckKeyPlace("output_coco_man2", "man")
ckp=CheckKeyPlace("output_coco_toilet", "toilet")
ckp=CheckKeyPlace("output_coco_toilet2", "toilet")


plt.figure(figsize=(20,12))
plt.xticks(fontsize=30)
plt.xticks(rotation=30)
plt.yticks(fontsize=30)
plt.subplots_adjust(bottom=0.15)
plt.boxplot(hakohige_lawdata,labels=hakohige_labels)
plt.savefig("keyplace.eps")
plt.figure(figsize=(20,12))
plt.xticks(fontsize=30)
plt.xticks(rotation=30)
plt.yticks(fontsize=30)
plt.subplots_adjust(bottom=0.15)
plt.boxplot(hakohige_data,labels=hakohige_labels)
plt.savefig("keyplace_nomarized.eps")