import pandas as pd

def PairConcatenate(samePairfilePath, datasetFilepath, outputFilePath,gsc =False):
    smPair_DF = pd.read_csv(samePairfilePath)
    hof_DF = pd.read_csv(datasetFilepath)
    finalList = []
    imageList = list()
    featuresList = pd.DataFrame()
    count = 1
    for index, row in smPair_DF.iterrows():
        imageidA = row['img_id_A']
        imageidB = row['img_id_B']
        targetValue = str(row['target'])
        if gsc:

            hofimageADataA = hof_DF.loc[hof_DF['img_id'] == imageidA].iloc[0, 1:].values.T.tolist()
            hofimageADataB = hof_DF.loc[hof_DF['img_id'] == imageidB].iloc[0, 1:].values.T.tolist()
        else:
            hofimageADataA = hof_DF.loc[hof_DF['img_id'] == imageidA].iloc[0, 2:].values.T.tolist()
            hofimageADataB = hof_DF.loc[hof_DF['img_id'] == imageidB].iloc[0, 2:].values.T.tolist()


        imageList.append(imageidA)
        imageList.append(imageidB)
        imageList += hofimageADataB + hofimageADataA
        imageList.append(targetValue)
        finalList.append(imageList)
       # featuresList = pd.DataFrame(finalList)
        imageList = []
        print('concat', count)
        count = count + 1
    featuresList = pd.DataFrame(finalList)
    featuresList.to_csv(outputFilePath, index=False)

def PairSubtract(samePairfilePath, datasetFilepath, outputFilePath,gsc= False):
    smPair_DF = pd.read_csv(samePairfilePath)
    hof_DF = pd.read_csv(datasetFilepath)
    finalList = []
    imageList = list()
    count = 1
    for index, row in smPair_DF.iterrows():
        imageidA = row['img_id_A']
        imageidB = row['img_id_B']
        targetValue = str(row['target'])
        if gsc:

            hofimageADataA = hof_DF.loc[hof_DF['img_id'] == imageidA].iloc[0, 1:]
            hofimageADataB = hof_DF.loc[hof_DF['img_id'] == imageidB].iloc[0, 1:]
        else:
            hofimageADataA = hof_DF.loc[hof_DF['img_id'] == imageidA].iloc[0, 2:]
            hofimageADataB = hof_DF.loc[hof_DF['img_id'] == imageidB].iloc[0, 2:]
        subFeatures = hofimageADataA.sub(hofimageADataB).abs()
        imageList.append(imageidA)
        imageList.append(imageidB)
        imageList += subFeatures.values.T.tolist()
        imageList.append(targetValue)
        finalList.append(imageList)
        #featuresList = pd.DataFrame(finalList)
        imageList = []
        print('subtract', count)

        count = count + 1
    featuresList = pd.DataFrame(finalList)
    featuresList.to_csv(outputFilePath, index=False)


def concatenateHOFCSVFiles(filepath1, filepath2, axisvalue):
    diffPair_DF = pd.read_csv(filepath2, nrows=791)
    smPair_DF = pd.read_csv(filepath1)
    df = pd.concat([diffPair_DF, smPair_DF])
    df.to_csv('combined_HOF.csv', index=False)

def concatenateGSCCSVFiles(filepath1, filepath2, axisvalue):
    diffPair_DF = pd.read_csv(filepath2)
    smPair_DF = pd.read_csv(filepath1)
    smallSmlDF = smPair_DF.sample(n= 10000)
    smallDiffDF = diffPair_DF.sample(n=10000)
    df = pd.concat([smallSmlDF, smallDiffDF])
    df.to_csv('combined_GSC.csv', index=False)




if __name__ == '__main__':
    concatenateGSCCSVFiles('//Users//aman//Desktop//GSC-Dataset//GSC-Features-Data//same_pairs.csv','//Users//aman//Desktop//GSC-Dataset//GSC-Features-Data//diffn_pairs.csv',0)
    concatenateHOFCSVFiles('//Users//aman//Desktop//same_pairs.csv', '//Users//aman//Desktop//diffn_pairs.csv', 0)
    # GSC
    PairConcatenate('combined_GSC.csv', '//Users//aman//Desktop//GSC-Dataset//GSC-Features-Data//GSC-Features.csv', 'concatenate_GSC.csv',True)
    PairSubtract('combined_GSC.csv', '//Users//aman//Desktop//GSC-Dataset//GSC-Features-Data//GSC-Features.csv', 'sub_GSC.csv',True)
    # HOF
    PairSubtract('combined_HOF.csv', '//Users//aman//Desktop//hof.csv', 'sub_HOF.csv')
    PairConcatenate('combined_HOF.csv', '//Users//aman//Desktop//hof.csv', 'concatenate_HOF.csv')
