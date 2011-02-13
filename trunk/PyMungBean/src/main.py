'''
Created on Feb 8, 2011

@author: anol
'''
import feature
import glob
import csv


if __name__ == '__main__':
    files = glob.glob('../image/10022011/*.jpg')

#===============================================================================
# Feature Extraction
#===============================================================================
    feature1 = feature.extraction.first_order_stat(files)
    feature2 = feature.extraction.moment_base(files)
    feature3 = feature.extraction.fourier(files)

    with open('features.csv', "w") as fd:
        write = csv.writer(fd, delimiter = '\t', quotechar = '|', quoting = csv.QUOTE_MINIMAL)
        header = feature1.values()[0].keys() + feature2.values()[0].keys()
        fd.write('file name\t')
        write.writerow(header)

        for f in files:
            fd.write(f + '\t')
            values = feature1[f].values() + feature2[f].values() + feature3[f].values()
            write.writerow(values)

#===============================================================================
# Feature Selection
#===============================================================================

#===============================================================================
# Machine Learning
#===============================================================================

#===============================================================================
# Classifier
#===============================================================================

