# coding = UTF-8
import random
import copy
import math
import numpy
import matplotlib.pyplot as plt
import sys

'''道路相关约束参数'''
n = 7  # 交叉口数目
block = [4, 4, 3, 5, 3, 4, 3] # 每个交叉口相位数
# xcount = 26  # 交叉口数目*交叉口相位数(D维空间）
xcount = 2 * n  # 1（C）+交叉口数目（协调相位绿信比）+交叉口数目-1（相位差数）

minPhase = [[35,35,8,10],[8,35,8,35],[10,35,35],[10,35,10,8,35],[10,35,35],[10,35,10,35],[10,35,35]] # 最小相位时长
minOth = [53,51,45,63,45,55,45] # 非协调相位最小相位总时长
basePhase = [[37,111,9,43],[28,121,13,38],[28,134,38],[34,75,31,19,41],[48,89,63],[35,67,52,46],[37,111,9,43]]
basePhaseDiff = [47,0,0,124,171,147] # 基准配时相位差方案
basePhase1 = [[40,120,10,30],[20,130,10,40],[20,140,40],[20,100,20,14,46],[40,100,60],[30,100,20,50],[28,112,60]]
basePhaseDiff1 = [55,0,199,123,171,177]
minCommonC = 98 # 最小周期时长
maxCommonC = 300 # 最大周期时长
minPhaseDiff = 0 # 最小相邻相位差(始终取正值）
maxPhaseDiff = maxCommonC # 最大相邻相位差


'''微粒群迭代参数'''

maxTurns = 500  # 最大迭代次数
EndCount = 0  # 连续迭代多次没有出现更优结果

Scales = 200  # 微粒群种群规模(N)
pos = []  # 每个微粒当前位置的集合
speed = []  # 每个微粒当前速度的集合
pbest = []  # 每个微粒自身最优历史位置的集合
gbest = []  # 所有微粒中的最优历史位置
w = 0.8  # 速度惯性权重
c1 = 2  # 学习因子/加速常数(认知）
c2 = 2  # 学习因子/加速常数(社会）
r1 = random.uniform(0, 1)  # 0-1均匀分布随机数(用来保证种群多样性)
r2 = random.uniform(0, 1)  # 0-1均匀分布随机数(用来保证种群多样性)


'''退火迭代参数'''
T0 = 0.1  # 初始温度
T_min = 0.06  # 冷却条件
cl = 0.99  # 冷却系数
cf = 0.02  # 快速冷却系数
TL = 1  # 每个T值下的迭代次数为TL


ans = 0  # 最后求的的全局最优解

'''文件读写'''
fr1=open("./input/result.txt","r")
fr2=open("./input/result2.txt","r")
fw1=open("./ans.txt","w")

g_upQ = []
g_downQ = []
g_dist = [637.789591088,428.654872829,399.577902292,814.455991752,978.273287482,900.826634264]
g_upV = []
g_upmaxv = 0
g_downV = []
g_downmaxv = 0
g_fullQ = []
g_Q = []

''' initial all Scales' pos,speed '''


def GenerateBase(plist):
    plist.append(200)
    for i in range(n):
        plist.append(basePhase[i][1]/float(plist[0]))
    for i in range(n-1):
        plist.append(basePhaseDiff[i])

def GenerateBase1(plist):
    plist.append(200)
    for i in range(n):
        plist.append(basePhase1[i][1]/float(plist[0]))
    for i in range(n-1):
        plist.append(basePhaseDiff1[i])

def GenerateRandPos(list):
    list.append(random.randint(minCommonC,maxCommonC))
    tempC=list[0]
    for i in range(n):
        list.append(random.uniform(float(minPhase[i][1])/float(tempC), float(tempC-minOth[i])/float(tempC)))
    for i in range(n-1):
        list.append(random.randint(minPhaseDiff,tempC))

def GenerateRandVel(list):
    list.append(random.uniform(minCommonC-maxCommonC,maxCommonC-minCommonC))
    for i in range(n):
        list.append(random.uniform(minOth[i]/float(maxCommonC)+minPhase[i][1]/float(maxCommonC)-1, 1-minOth[i]/float(maxCommonC)-minPhase[i][1]/float(maxCommonC)))
    for i in range(n-1):
        list.append(random.uniform(minPhaseDiff-maxPhaseDiff,maxPhaseDiff-minPhaseDiff))
'''
  计算种群中粒子的适应度(目标函数）
  首先针对单个交叉口计算
  之后根据干线协调控制原理重新调整周期与相位时长
'''
def readFiles():
    global g_upmaxv,g_downmaxv
    xp=0
    yp=0
    f1_lines=fr1.readlines()
    for line in f1_lines:
        line=line.strip()
        if(line.isdigit()):
            xp=int(line)
            g_fullQ.append([])
            g_Q.append([])
        else:
            tmp=line.split()
            yp=int(tmp[0])
            g_fullQ[xp].append(float(tmp[2]))
            g_Q[xp].append(float(tmp[1]))

    '''调整第一个路口相位的特殊性'''
    pos =len(g_Q[0]) - 1
    tmp1=g_Q[0][pos]
    tmp2=g_fullQ[0][pos]
    while pos>0:
        g_Q[0][pos]=g_Q[0][pos-1]
        g_fullQ[0][pos]=g_fullQ[0][pos-1]
        pos-=1
    g_Q[0][0]=tmp1
    g_fullQ[0][0]=tmp2

    f2_lines=fr2.readlines()
    for line in f2_lines:
        line=line.strip()
        if(line.isdigit()):
            xp=int(line)
            yp=0
        else:
            tmp=line.split()
            if(yp==0):
                '''up'''
                g_upQ.append(float(tmp[1]))
                g_upV.append(float(tmp[2]))
                yp=1
            else:
                '''down'''
                g_downQ.append(float(tmp[1]))
                g_downV.append(float(tmp[2]))
                yp=0
    g_upmaxv=max(g_upV)
    g_downmaxv=max(g_downV)


def getUpQ():
    # 每个交叉口协调相位的上行流量 l/s
    # read file
    return g_upQ


def getDownQ():
    # 每个交叉口协调相位的下行流量 l/s
    return g_downQ


def getDist():
    # 相邻两个交叉口之间的距离 m
    return g_dist


def getUpV():
    # 每个交叉口协调相位的上行速度 m/s
    # TODO:考虑Vup是上行中的最大速度还是每一段的速度
    # 后续实现中Vup是认为为一个值，也就是最大速度/平均速度
    # 如果要为每一段的速度哦相应代码需要更改
    # return [11.1,8.3,5.5,6.0,6.5,8.3,12]
    return g_upmaxv


def getDownV():
    # 每个交叉口协调相位的下行速度 m/s
    # TODO:考虑Vdown是下行中的最大速度还是每一段的速度
    # 后续实现中Vdown是认为为一个值，也就是最大速度
    # 如果要为每一段的速度相应代码需要更改
    # return [11.1,8.3,5.5,6.0,6.5,8.3,12]
    return g_downmaxv


def getFullQ():
    # 每个交叉口每个相位的饱和流量 l/s
    return g_fullQ


def getQ():
    # 每个交叉口每个相位的实际流量 l/s
    return g_Q


def getAbility(C,fulQ, lamda):
    # 每个交叉口协调相位的通行能力
    Ai = []
    for i in range(n):
        # Ai.append(fulQ[i][1] * lamda[i][1])
        Ai.append(fulQ[i][1])
    return Ai


def getLamda(list, q):
    lamda = []
    Qi = []
    for i in range(n):
        Qi.append(0)
        for j in range(block[i]):
            if j != 1:
                Qi[i] += q[i][j]
    for i in range(n):
        lamda.append([])
        for j in range(block[i]):
            if j == 1:
                lamda[i].append(list[i + 1])
            else:
                lamda[i].append((1 - list[i + 1]) * q[i][j] / Qi[i])
    return lamda


def isUpBlocked(l, Vup, list):
    # 每个交叉口协调相位上行车流是全受阻还是部分受阻
    alphai = []
    for i in range(n-1):
        if math.ceil(l[i] / Vup) % list[0] <= list[n + 1 + i]:
            alphai.append(1)
        else:
            alphai.append(0)

    return alphai


def isDownBlocked(l, Vdown, list):
    # 每个交叉口协调相位下行车流是全受阻还是部分受阻
    betai = []
    for i in range(n-1):
        if list[0] - math.ceil(l[i] / Vdown) % list[0] >= list[n + 1 + i]:
            betai.append(1)
        else:
            betai.append(0)
    return betai


def calCoPhase(o2, l, Vup, Vdown, list, Qup, Qdown, Mui, lamda,Muik,isp=False):
    sum = 0
    alphai = []  # i个值,0/1,第i个交叉口(上行)全部受阻取1,部分受阻取0
    betai = []  # i个值,0/1,第i个交叉口(下行)全部受阻取1,部分受阻取0
    alphai = isUpBlocked(l, Vup, list)
    betai = isDownBlocked(l, Vdown, list)
    # 注意这里没有关注边缘交叉口的本部分延误，包括下到上进入区域到达第一个交叉口的上行延误，以及从上倒下进入区域到达最后一个交叉口的下行延误
    # 可以认为边缘延误具有随机性，在计算万中间部分之后，加上边缘随机延误
    # 注意所有上下都是相对的，上即指向路口标号更大的方向行驶，上行指从路口i向路口i+1的方向
    for i in range(n-1):
        Diup = alphai[i] * Qup[i+1] * Mui[i+1] * pow((list[n + 1 + i] - math.ceil(l[i] / Vup) % list[0]), 2) / (
        2 * (Mui[i+1] - Qup[i+1])) + (1 - alphai[i]) * pow(list[0] * (1 - lamda[i][1]), 2) * Qup[i+1] * Mui[i+1] / (
        2 * (Mui[i+1] - Qup[i+1]))
        Didown = betai[i] * Qdown[i] * Mui[i] * pow(list[0] - list[n + 1 + i] - math.ceil(l[i] / Vdown) % list[0],
                                                    2) / (2 * (Mui[i] - Qdown[i])) + (1 - betai[i]) * pow(
            list[0] * (1 - lamda[i][1]), 2) * Qdown[i] * Mui[i] / (2 * (Mui[i] - Qdown[i]))
        sum = sum + o2 * Diup + (2 - o2) * Didown
        # TODO:如果通行能力小于到达流量，那么停在路口的车流量不会减少，只会持续增加，也就会产生彻底堵死？delay会变成负的
        # TODO:如果通行能力大于到的流量，但是绿灯时间内车流没有被完全疏散，则由一次完全受阻，进入下一次完全受阻，不影响

    if o2==0:
        #只考虑下行
        sum = sum + list[0] * Qdown[n-1] * (1 - lamda[n-1][1]) * (1 - lamda[n-1][1]) / (2 * (1 - lamda[n-1][1] * Muik[n-1][1]))
        + Muik[n-1][1] * Muik[n-1][1] / (2 * (1 - Muik[n-1][1]))
    elif o2==2:
        #只考虑上行
        sum = sum + list[0] * Qup[0] * (1 - lamda[0][1]) * (1 - lamda[0][1]) / (2 * (1 - lamda[0][1] * Muik[0][1]))
        + Muik[0][1] * Muik[0][1] / (2 * (1 - Muik[0][1]))
    else:
        #考虑双向
        sum = sum + list[0] * Qdown[n-1] * (1 - lamda[n-1][1]) * (1 - lamda[n-1][1]) / (2 * (1 - lamda[n-1][1] * Muik[n-1][1]))
        + Muik[n-1][1] * Muik[n-1][1] / (2 * (1 - Muik[n-1][1]))+list[0] * Qup[0] * (1 - lamda[0][1]) * (1 - lamda[0][1]) / (2 * (1 - lamda[0][1] * Muik[0][1]))
        + Muik[0][1] * Muik[0][1] / (2 * (1 - Muik[0][1]))
    return sum


def calOthPhase(C, q, lamda, Muik):
    sum = 0
    for i in range(n):
        for k in range(block[i]):
            if k != 1:
                sum = sum + (C * q[i][k] * (1 - lamda[i][k]) * (1 - lamda[i][k]) / (2 * (1 - lamda[i][k] * Muik[i][k]))
                             + Muik[i][k] * Muik[i][k] / (2 * (1 - Muik[i][k])))
    return sum

def calCoPhase1(C,lamda,q,fulQ):
    sum = 0
    for i in range(n):
        sum += C * pow((1.0 - lamda[i][1]),2) / (2*(1.0 - q[i][1] / fulQ[i][1]))
    return sum

def calOthPhase1(C,lamda,q,fulQ):
    sum = 0
    for i in range(n):
        for k in range(block[i]):
            if k != 1:
                sum += C * pow((1.0 - lamda[i][k]), 2) / (2*(1.0 - q[i][k] / fulQ[i][k]))
    return sum

def calCoStops(lamda, q, fulQ):
    sum = 0
    for i in range(n):
        sum += 0.9 * (1.0 - lamda[i][1]) / (1.0 - q[i][1] / fulQ[i][1])
    return sum


def calOthStops(lamda, q, fulQ):
    sum = 0
    for i in range(n):
        for k in range(block[i]):
            if k != 1:
                sum += 0.9 * (1.0 - lamda[i][k]) / (1.0 - q[i][k] / fulQ[i][k])
    return sum


def CalFitness(list,isp=False):
    # list长度为xcount(所有相位的时长[其实就是这个相位保持绿灯的时间]或者绿信比)+n-1(i交叉口到i+1交叉口的相位差）
    answer = 0.0

    ''' calculation framework for delay '''

    o1 = 0.6  # [0,1],衡量主次干道(协调/非协调)分配,0:只考虑非协调相位,1:只考虑协调相位
    o2 = 1  # {0,1,2},0:只考虑下行方向车辆延误;1:考虑双向车辆延误;2:只考虑上行方向车辆延误

    l = getDist()  # l[i]=dis(交叉口i，交叉口i+1）

    # 协调相位延迟计算相关参数
    Qup = getUpQ()  # Qup[i]=交叉口i的协调相位上行流量
    Qdown = getDownQ()  # Qdown[i]=交叉口i的协调相位下行流量
    Vup = getUpV()
    Vdown = getDownV()

    lamda = []  # 每个相位的绿信比

    # Ci = []  #周期
    # count = 0
    # for i in range(n):
    #     C = 0.0
    #     for k in range(block[i]):
    #         C=C+list[count]
    #         count=count+1
    #     Ci.append(C)
    # CommonC = max(Ci)
    # list = modifyList(list,CommonC)

    # 对于为最大周期的交叉口数据不不变,其余的保持原随机数据中协调相位的绿信比不变
    # (为保持新周期下的比例不变，数值肯定要变),非协调相位的数值根据剩余数值按照流量比例分配
    q = getQ()
    lamda = getLamda(list, q)

    fulQ = getFullQ()
    # 每个交叉口协调相位的通行能力？也就是需要饱和流量*协调相位绿信比
    Mui = getAbility(list[0],fulQ, lamda)
    Muik = []  # 每个交叉口每个相位的饱和度
    for i in range(len(fulQ)):
        Muik.append([])
        for k in range(block[i]):
            if k == 1:
                Muik[i].append((Qup[i] + Qdown[i]) / (fulQ[i][k] * lamda[i][k]))
            else:
                Muik[i].append(q[i][k] / fulQ[i][k] * lamda[i][k])

    Delay = o1 * calCoPhase(o2, l, Vup, Vdown, list, Qup, Qdown, Mui, lamda, Muik) + (1 - o1) * calOthPhase(list[0], q, lamda,
                                                                                          Muik)

    Stops = o1 * calCoStops(lamda, q, fulQ) + (1 - o1) * calOthStops(lamda, q, fulQ)

    answer = 1000 / (Delay + 10 * Stops)

    return answer

def CalFitness1(list):
    q = getQ()
    lamda = getLamda(list, q)

    fulQ = getFullQ()

    o1 = 0.6  # [0,1],衡量主次干道(协调/非协调)分配,0:只考虑非协调相位,1:只考虑协调相位

    Delay = o1 * calCoPhase1(list[0],lamda, q, fulQ) + (1 - o1) * calOthPhase1(list[0],lamda, q, fulQ)

    Stops = o1 * calCoStops(lamda, q, fulQ) + (1 - o1) * calOthStops(lamda, q, fulQ)

    answer = 1000 / (Delay + 10 * Stops)

    return answer


''' calculate gbest '''


def FindSwarmsMostPos():
    best = CalFitness(pbest[0])
    # best = CalFitness1(pbest[0])
    index = 0
    for i in range(Scales-1):
        temp = CalFitness(pbest[i+1])
        # temp = CalFitness1(pbest[i+1])
        if temp > best:
            best = temp
            index = i
    return pbest[index]


''' define vector calculation functions '''


def NumMulVec(num, list):
    # result is in list
    for i in range(len(list)):
        list[i] *= num
    return list


def VecSubVec(list1, list2):
    # result is in list1
    for i in range(len(list1)):
        list1[i] -= list2[i]
    return list1


def VecAddVec(list1, list2):
    # result is in list1
    for i in range(len(list1)):
        list1[i] += list2[i]
    return list1


''' define parameters' updating functions '''


def UpdateSpeed():
    # global speed
    # Check whether r1,r2 need change in each update operation or not
    # Check Result: Need to change r1,r2 within every operation
    r1 = random.uniform(0, 1)
    r2 = random.uniform(0, 1)
    for i in range(Scales):
        temp1 = NumMulVec(w, speed[i][:])
        temp2 = VecSubVec(pbest[i][:], pos[i])
        temp2 = NumMulVec(c1 * r1, temp2[:])
        temp1 = VecAddVec(temp1[:], temp2)
        temp2 = VecSubVec(gbest[:], pos[i])
        temp2 = NumMulVec(c2 * r2, temp2[:])
        speed[i] = VecAddVec(temp1, temp2)
        checkSpeed(speed[i])

def UpdatePos():
    global gbest
    for i in range(Scales):
        VecAddVec(pos[i], speed[i])
        checkPos(pos[i])
        if CalFitness(pos[i]) < CalFitness(pbest[i]):
        # if CalFitness(pos[i]) > CalFitness(pbest[i]):
        # if CalFitness1(pos[i]) > CalFitness1(pbest[i]):
            pbest[i] = copy.deepcopy(pos[i])
    gbest = FindSwarmsMostPos()
    # gbest = SAO()

def checkSpeed(list):
    # 需要满足speed不大于pos允许搜索的区间宽度，不小于负的区间宽度
    rt=True

    limit0=maxCommonC-minCommonC
    if list[0]>limit0:
        list[0]=limit0
        rt=False
    elif list[0]<-limit0:
        list[0]=-limit0
        rt=False

    for i in range(n):
        tmpLimit=1-minOth[i]/float(maxCommonC)-minPhase[i][1]/float(maxCommonC)
        if list[i+1]>tmpLimit:
            list[i+1]=tmpLimit
            rt = False
        elif list[i+1]<-tmpLimit:
            list[i+1]=-tmpLimit
            rt = False

    for i in range(n-1):
        tmpLimit=maxPhaseDiff-minPhaseDiff
        if list[n+1+i]>tmpLimit:
            list[n+1+i]=tmpLimit
            rt = False
        elif list[n+1+i]<-tmpLimit:
            list[n+1+i]=-tmpLimit
            rt = False

    return rt

def checkPos(list):

    rt=True
    if list[0]>maxCommonC:
        list[0]=maxCommonC
        rt=False
    elif list[0]<minCommonC:
        list[0]=minCommonC
        rt=False

    for i in range(n):
        tmp=list[i+1]*list[0]
        tmpLimit1=minPhase[i][1]
        tmpLimit2=list[0]-minOth[i]
        if tmp<tmpLimit1:
            list[i+1]=minPhase[i][1]/float(list[0])
            rt=False
        if tmp>tmpLimit2:
            list[i+1]=1-minOth[i]/float(list[0])
            rt=False

    for i in range(n-1):
        if list[n+1+i]>list[0] or list[n+1+i]<0:
            list[n+1+i]=list[n+1+i]%list[0]
            rt=False

    return rt

def RandUpdateGbest():

    #方案1
    # global gbest
    # n_gbest = copy.deepcopy(gbest)
    # rand_vec = []
    # GenerateRandVel(rand_vec)  #全局搜索半径
    # randw=random.uniform(0,1)  #搜索半径缩放
    # temp1=NumMulVec(randw,rand_vec)
    # temp2=VecAddVec(n_gbest,temp1)
    # new_gbest=copy.deepcopy(temp2)

    #方案2
    new_gbest=[]
    GenerateRandPos(new_gbest)
    return new_gbest

def SAO():
    T = T0
    nbest = RandUpdateGbest()
    tbest = copy.deepcopy(gbest)
    while T > T_min:
        for i in range(TL):
            df=CalFitness(nbest)-CalFitness(tbest)
            # df=CalFitness1(nbest)-CalFitness1(tbest)
            # if df > 0:
            if df < 0:
                tbest = copy.deepcopy(nbest)
            else:
                r = random.uniform(0, 1)
                if math.exp(df / T) > r:
                    # TODO:df<0时，df/T太接近0导致取较差值的概率非常大
                    tbest = copy.deepcopy(nbest)
            nbest = RandUpdateGbest()
        T = T * cl
    return tbest

def FSAO():
    T = T0
    nbest = RandUpdateGbest()
    tbest = copy.deepcopy(gbest)
    t = 1
    cnt = 0
    while T > T_min:
        temp = copy.deepcopy(tbest)

        for i in range(TL):
            df = CalFitness(nbest) - CalFitness(tbest)
            # df = CalFitness1(nbest) - CalFitness1(tbest)
            # if df > 0:
            if df > 0:
                tbest = copy.deepcopy(nbest)
            else:
                if math.exp(df / T) > random.uniform(0, 1):
                    tbest = copy.deepcopy(nbest)
            nbest = RandUpdateGbest()

        ddf = CalFitness(tbest) - CalFitness(temp)
        # ddf = CalFitness1(tbest) - CalFitness1(temp)

        # if ddf <= 0:
        if ddf >= 0:
            cnt += 1
        else:
            cnt = 0

        if cnt >= 5:
            T = T * (1 + cf)
        else:
            T = T / (1 + cf * t)

        t += 1

    return tbest

if __name__ == '__main__':
    readFiles()
    ''' initial data structure '''
    for i in range(Scales):
        pos.append([])
        speed.append([])
        pbest.append([])

    ''' initial swarm first generation '''
    a_s=int(Scales/2)
    b_s=int(Scales/4)
    c_s=Scales-a_s-b_s
    for i in range(a_s):
        GenerateRandPos(pos[i])
        GenerateRandVel(speed[i])
        pbest[i] = copy.deepcopy(pos[i])
    for i in range(b_s):
        GenerateBase(pos[a_s+i])
        GenerateRandVel(speed[a_s+i])
        pbest[a_s+i] = copy.deepcopy(pos[a_s+i])
    for i in range(c_s):
        GenerateBase1(pos[a_s+b_s+i])
        GenerateRandVel(speed[a_s+b_s+i])
        pbest[a_s+b_s+i] = copy.deepcopy(pos[a_s+b_s+i])

    # gbest0=[200,0.735,0.72,0.775,0.68,0.775,0.725,0.74,106,109,93,108,108,106]
    # gbest1=[]
    # GenerateBase(gbest1)
    # gbest2=[]
    # GenerateBase1(gbest2)
    # gbest3=[98,0.4489796,0.4285714,0.5408163,0.3571429,0.5408163,0.3571429,0.4693878,50,73,74,64,72,44]
    # ans0 = CalFitness(gbest0)
    # ans1 = CalFitness1(gbest0)
    # ans2 = CalFitness(gbest1)
    # ans3 = CalFitness1(gbest1)
    # ans4 = CalFitness(gbest2)
    # ans5 = CalFitness1(gbest2)
    # ans6 = CalFitness(gbest3)
    # ans7 = CalFitness1(gbest3)
    # print("ans0:%f,ans1:%f,1.4592"%(ans0,ans1))
    # print("ans6:%f,ans7:%f,1.7214" % (ans6, ans7))
    # print("ans4:%f,ans5:%f,2.7813"%(ans4,ans5))
    # print("ans2:%f,ans3:%f,2.9856" % (ans2, ans3))
    gbest = FindSwarmsMostPos()
    # gbest = SAO()
    ans = CalFitness(gbest)
    # ans = CalFitness1(gbest)

    turns = []
    fitness = []

    ''' loops to update generations '''
    for i in range(maxTurns):
        turns.append(i+1)
        fitness.append(ans)
        print('Turns: %d'%(i+1))
        fw1.write('Turns: %d, Fitness: %f.\n'%((i+1),ans))
        fw1.write(','.join(str(i) for i in gbest))
        fw1.write('\n')
        UpdateSpeed()
        UpdatePos()
        cur_ans = CalFitness(gbest)
        # cur_ans = CalFitness1(gbest)
        if (cur_ans < ans):
            ans = cur_ans
            EndCount = 0
            continue
        else:
            EndCount = EndCount + 1
            # if (EndCount > 5):
            #     print('Don\'increase anymore! Loops terminate.')
            #     break


    # plt.plot(turns, fitness, 'b*')  # ,label=$cos(x^2)$)
    plt.plot(turns, fitness, 'r')
    plt.xlabel('turns')
    plt.ylabel('fitness')
    plt.title('Fitness-Turn diagram')
    plt.legend()
    plt.show()
    fr1.close()
    fr2.close()
    fw1.flush()
    fw1.close()
    sys.exit(0)