import ahocorasick
from elmoformanylangs import Embedder
import jieba
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 读取数据
def read_data(file_path):
    with open(file_path,encoding='utf-8') as file:
        spos = {}
        for line in file.readlines():
            spo = line.strip().split('\t')
            s = spo[0][1:-1]
            p = spo[1][1:-1]
            o = spo[2][1:-3]
            if s not in spos:
                spos[s] = []
            spos[s].append((p,o))
    return spos

# 构建ac自动机
def build_ac(data):
    ac_tree = ahocorasick.Automaton()
    for k, v in data.items():
        ac_tree.add_word(k, v)
    ac_tree.make_automaton()
    return ac_tree

# 转换句向量
def elmo2vec(text):
    res = list(jieba.cut(text))
    vec = model.sents2elmo([res])   # [n_sent,n_token,1024]
    # 得到句子向量
    vec = [np.mean(sent, axis=0) for sent in vec]  # [n_sent,1024]
    return vec

# 计算两个句子的余弦相似度
def cosSim(a,b):
    return np.matmul(a,np.array(b).T)/np.linalg.norm(a)/np.linalg.norm(b)

# 计算问题的最佳答案
def maxSim(text):
    sim_list = []
    text_embed = elmo2vec(text)
    res = ac_tree.iter(text)
    for index,value in res:     # 可能找到多个subject
        cur_sim = []
        for v in value:
            predict_embed = elmo2vec(v[0])
            cur_sim.append(cosSim(text_embed,predict_embed))
        if max(cur_sim) < 0.25:
            print('没找到')
            break
        max_index = np.argmax(cur_sim)
        sim_list.append((index,value[max_index]))
        print(sim_list)
    return sim_list

spos = read_data('./data/data')

ac_tree = build_ac(spos)

model = Embedder('./elmo')

text = '卧佛寺乡的下辖地区'

maxSim(text)
