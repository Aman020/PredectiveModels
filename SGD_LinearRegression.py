from sklearn.cluster import KMeans
import numpy as np
import csv
import math
import matplotlib.pyplot as plt

maxAcc = 0.0
maxIter = 0
C_Lambda = 0.05
TrainingPercent = 80
ValidationPercent = 10
TestPercent = 10
M = 20
PHI = []
IsSynthetic = False

def GetTargetVector(filePath):
    t = []
    with open(filePath, 'rU') as f:
        reader = csv.reader(f)
        for row in reader:
            t.append(int(row[20]))
    #print("Raw Training Generated..")
    return t[1:]


def GenerateRawData(filePath, IsSynthetic):
    dataMatrix = []
    with open(filePath, 'rU') as fi:
        reader = csv.reader(fi)
        next(reader)
        for row in reader:
            row = row[2:]
            dataRow = []
            for column in row:
                dataRow.append(float(column))
            dataMatrix.append(dataRow)
    dataMatrix = np.transpose(dataMatrix)
    # print ("Data Matrix Generated..")
    return dataMatrix

def GenerateTrainingTarget(rawTraining,TrainingPercent = 80):
    TrainingLen = int(math.ceil(len(rawTraining)*(TrainingPercent*0.01)))
    t           = rawTraining[:TrainingLen]
    #print(str(TrainingPercent) + "% Training Target Generated..")
    return t


def GenerateTrainingDataMatrix(rawData, TrainingPercent = 80):
    T_len = int(math.ceil(len(rawData[0])*0.01*TrainingPercent))
    d2 = rawData[:,0:T_len]
    #print(str(TrainingPercent) + "% Training Data Generated..")
    return d2

def GenerateValData(rawData, ValPercent, TrainingCount):
    valSize = int(math.ceil(len(rawData[0])*ValPercent*0.01))
    V_End = TrainingCount + valSize
    # Splicing the raw data and getting only the required data
    dataMatrix = rawData[:,TrainingCount+1:V_End]
    #print (str(ValPercent) + "% Val Data Generated..")
    return dataMatrix

def GenerateValTargetVector(rawData, ValPercent, TrainingCount):
    valSize = int(math.ceil(len(rawData)*ValPercent*0.01))
    V_End = TrainingCount + valSize
    t =rawData[TrainingCount+1:V_End]
    #print (str(ValPercent) + "% Val Target Data Generated..")
    return t


def GenerateBigSigma(Data, MuMatrix, TrainingPercent, IsSynthetic):
    BigSigma = np.zeros((len(Data), len(Data)))
    DataT = np.transpose(Data)
    TrainingLen = math.ceil(len(DataT) * (TrainingPercent * 0.01))
    varVect = []
    for i in range(0, len(DataT[0])):
        vct = []
        for j in range(0, int(TrainingLen)):
            vct.append(Data[i][j])
        varVect.append(np.var(vct))

    for j in range(len(Data)):
        BigSigma[j][j] = varVect[j]
    if IsSynthetic == True:
        BigSigma = np.dot(3, BigSigma)
    else:
        BigSigma = np.dot(200, BigSigma)
    ##print ("BigSigma Generated..")
    return BigSigma

def GetScalar(DataRow,MuRow, BigSigInv):
    R = np.subtract(DataRow,MuRow)
    T = np.dot(BigSigInv,np.transpose(R))
    L = np.dot(R,T)
    return L

def GetRadialBasisOut(DataRow,MuRow, BigSigInv):
    phi_x = math.exp(-0.5*GetScalar(DataRow,MuRow,BigSigInv))
    return phi_x

def GetPhiMatrix(Data, MuMatrix, BigSigma, TrainingPercent = 80):
    DataT = np.transpose(Data)
    TrainingLen = math.ceil(len(DataT)*(TrainingPercent*0.01))
    PHI = np.zeros((int(TrainingLen),len(MuMatrix)))
    BigSigInv = np.linalg.inv(BigSigma)
    for  C in range(0,len(MuMatrix)):
        for R in range(0,int(TrainingLen)):
            PHI[R][C] = GetRadialBasisOut(DataT[R], MuMatrix[C], BigSigInv)
    #print ("PHI Generated..")
    return PHI

def GetWeightsClosedForm(PHI, T, Lambda):
    Lambda_I = np.identity(len(PHI[0]))
    for i in range(0,len(PHI[0])):
        Lambda_I[i][i] = Lambda
    PHI_T       = np.transpose(PHI)
    PHI_SQR     = np.dot(PHI_T,PHI)
    PHI_SQR_LI  = np.add(Lambda_I,PHI_SQR)
    PHI_SQR_INV = np.linalg.inv(PHI_SQR_LI)
    INTER       = np.dot(PHI_SQR_INV, PHI_T)
    W           = np.dot(INTER, T)
    ##print ("Training Weights Generated..")
    return W

def GetPhiMatrix(Data, MuMatrix, BigSigma, TrainingPercent = 80):
    DataT = np.transpose(Data)
    TrainingLen = math.ceil(len(DataT)*(TrainingPercent*0.01))
    PHI = np.zeros((int(TrainingLen),len(MuMatrix)))
    BigSigInv = np.linalg.inv(BigSigma)
    for  C in range(0,len(MuMatrix)):
        for R in range(0,int(TrainingLen)):
            PHI[R][C] = GetRadialBasisOut(DataT[R], MuMatrix[C], BigSigInv)
    #print ("PHI Generated..")
    return PHI

def GetValTest(VAL_PHI,W):
    Y = np.dot(W,np.transpose(VAL_PHI))
    ##print ("Test Out Generated..")
    return Y

def GetErms(VAL_TEST_OUT,ValDataAct):
    sum = 0.0
    t=0
    accuracy = 0.0
    counter = 0
    val = 0.0
    for i in range (0,len(VAL_TEST_OUT)):
        sum = sum + math.pow((ValDataAct[i] - VAL_TEST_OUT[i]),2)
        if(int(np.around(VAL_TEST_OUT[i], 0)) == ValDataAct[i]):
            counter+=1
    accuracy = (float((counter*100))/float(len(VAL_TEST_OUT)))
    ##print ("Accuracy Generated..")
    ##print ("Validation E_RMS : " + str(math.sqrt(sum/len(VAL_TEST_OUT))))
    return (str(accuracy) + ',' +  str(math.sqrt(sum/len(VAL_TEST_OUT))))


RawTarget = GetTargetVector('concatenate_HOF.csv')
RawData   = GenerateRawData('concatenate_HOF.csv',IsSynthetic)

TrainingTarget = np.array(GenerateTrainingTarget(RawTarget,TrainingPercent))
TrainingData   = GenerateTrainingDataMatrix(RawData,TrainingPercent)
print(TrainingTarget.shape)
print(TrainingData.shape)

ValDataAct = np.array(GenerateValTargetVector(RawTarget,ValidationPercent, (len(TrainingTarget))))
ValData    = GenerateValData(RawData,ValidationPercent, (len(TrainingTarget)))
print(ValDataAct.shape)
print(ValData.shape)

TestDataAct = np.array(GenerateValTargetVector(RawTarget,TestPercent, (len(TrainingTarget)+len(ValDataAct))))
TestData = GenerateValData(RawData,TestPercent, (len(TrainingTarget)+len(ValDataAct)))
print(ValDataAct.shape)
print(ValData.shape)

ErmsArr = []
AccuracyArr = []

# Creating clusters and finding the centroid in each cluster.
kmeans = KMeans(n_clusters=M, random_state=0).fit(np.transpose(TrainingData))
Mu = kmeans.cluster_centers_

BigSigma     = GenerateBigSigma(RawData, Mu, TrainingPercent,IsSynthetic)
TRAINING_PHI = GetPhiMatrix(RawData, Mu, BigSigma, TrainingPercent)
W            = GetWeightsClosedForm(TRAINING_PHI,TrainingTarget,(C_Lambda))
TEST_PHI     = GetPhiMatrix(TestData, Mu, BigSigma, 100)
VAL_PHI      = GetPhiMatrix(ValData, Mu, BigSigma, 100)

W_Now = np.dot(220, W)
La = 2
learningRate = 0.04
L_Erms_Val = []
L_Erms_TR = []
L_Erms_Test = []
W_Mat = []

# This loop is to iterate over the phi matrix until convergence
for i in range(0, 500):
    # print ('---------Iteration: ' + str(i) + '--------------')
    Delta_E_D = -np.dot((TrainingTarget[i] - np.dot(np.transpose(W_Now), TRAINING_PHI[i])), TRAINING_PHI[i])
    La_Delta_E_W = np.dot(La, W_Now)
    Delta_E = np.add(Delta_E_D, La_Delta_E_W)
    Delta_W = -np.dot(learningRate, Delta_E)
    W_T_Next = W_Now + Delta_W
    W_Now = W_T_Next

    # -----------------TrainingData Accuracy---------------------#
    TR_TEST_OUT = GetValTest(TRAINING_PHI, W_T_Next)
    Erms_TR = GetErms(TR_TEST_OUT, TrainingTarget)
    L_Erms_TR.append(float(Erms_TR.split(',')[1]))

    # -----------------ValidationData Accuracy---------------------#
    VAL_TEST_OUT = GetValTest(VAL_PHI, W_T_Next)
    Erms_Val = GetErms(VAL_TEST_OUT, ValDataAct)
    L_Erms_Val.append(float(Erms_Val.split(',')[1]))

    # -----------------TestingData Accuracy---------------------#
    TEST_OUT = GetValTest(TEST_PHI, W_T_Next)
    Erms_Test = GetErms(TEST_OUT, TestDataAct)
    L_Erms_Test.append(float(Erms_Test.split(',')[1]))

plt.plot(np.arange(500), L_Erms_Test)
plt.xlabel("Number of iterations")
plt.ylabel("ERMS -Test")
plt.show()



print ('----------Gradient Descent Solution--------------------')
print ("M = 15 \nLambda  = 0.0001\neta=0.01")
print ("E_rms Training   = " + str(np.around(min(L_Erms_TR),5)))
print ("E_rms Validation = " + str(np.around(min(L_Erms_Val),5)))
print ("E_rms Testing    = " + str(np.around(min(L_Erms_Test),5)))