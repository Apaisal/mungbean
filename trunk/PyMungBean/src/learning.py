'''
Created on Feb 7, 2012

@author: anol
'''
import PyML as ml
from PyML.classifiers import svm, multi

if __name__ == '__main__':
    #===============================================================================
# Machine Learning
#===============================================================================
    trainingset1 = ml.SparseDataSet("./selected.data")
    trainingset2 = ml.SparseDataSet("./selected.data")
#    trainingset1 = ml.VectorDataSet(selected_file, labelsColumn=0)
#    trainingset2 = ml.VectorDataSet(selected_file, labelsColumn=0)
    testset1 = ml.SparseDataSet("./test.data")
    testset2 = ml.SparseDataSet("./test.data")


#    testset1 = ml.VectorDataSet(test_file, labelsColumn=0)
#    testset2 = ml.VectorDataSet(test_file, labelsColumn=0)
#    classifier.decisionSurface(sl, trainingset1, testset1)

    k2 = ml.ker.Polynomial(3)
    k1 = ml.ker.Linear()

    snl1 = multi.OneAgainstRest(svm.SVM(\
                                      k2 , \
                                      c = 10, \
#                                      optimizer = 'mysmo' \
                                      ))
#    snl2 = multi.OneAgainstOne(svm.SVM(\
#                                      k2 , \
#                                      c = 10, \
##                                      optimizer = 'mysmo' \
#                                      ))
#    snl = ml.SVM(k2)
#    snl.C = 10
    sl1 = multi.OneAgainstRest(svm.SVM(\
                                      k1 , \
                                      c = 10, \
#                                      optimizer = 'mysmo' \
                                      ))
#    sl2 = multi.OneAgainstOne(svm.SVM(\
#                                      k1 , \
#                                      c = 10, \
##                                      optimizer = 'mysmo' \
#                                      ))
#    sl = ml.SVM(k1)
#    sl.C = 10
#===============================================================================
# Linear Classifier
#===============================================================================
#    classifier.decisionSurface(sl, trainingset1, testset1)
    itert = 100
    rocN = 100
    numFold = 5
    normalize = True
    sl1.train(trainingset1)


    result1 = sl1.nCV(testset1, \
                     seed = 1, \
                      cvType = "stratifiedCV", \
#                      cvType = "cv", \
#                       intermediateFile = './result_linear' \
                     iterations = itert, \
                      numFolds = numFold)


    with open("./result_linear_iter1", "w") as fd:
        for res in result1:
            fd.write(str(res) + "\n")
#            res.plotROC("a.pdf", rocN = 100)
        fd.write(str(result1) + "\n")
        fd.close()
    result1.save("./linear_result1")


#===============================================================================
# Non Linear Classifier
#===============================================================================

    snl1.train(trainingset2)
#    snl2.train(trainingset2)

    result3 = snl1.nCV(testset2, \
                      seed = 1, \
#                      cvType = "cv", \
                      cvType = "stratifiedCV", \
#                      intermediateFile = './result_nonlinear', \
                      iterations = itert, \
                      numFolds = numFold)

    with open("result_nonlinear_iter1", "w") as fd:
        for res in result3:
            fd.write(str(res) + "\n")
        fd.write(str(result3) + "\n")
        fd.close()
    result3.save("./nonlinear_result1")

