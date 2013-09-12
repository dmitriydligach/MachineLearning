package em.experiments;

import java.io.IOException;
import java.util.HashSet;
import java.util.Random;

import cv.LearningCurve;
import data.Dataset;
import data.I2b2Dataset;
import data.Split;
import em.implementation.EmAlgorithm;

public class EmCurve {

  public static void main(String[] args) throws IOException {

    final int FOLDS = 5; // number of folds
    final int MAXLABELED = 100; // maximum number of labeled examples
        
    I2b2Dataset dataset = new I2b2Dataset();
    dataset.loadCSVFile("/home/dima/active/ibd/data/data.txt", "/home/dima/active/ibd/data/labels-cd.txt");
    dataset.makeAlphabets();

    Split[] splits = dataset.split(FOLDS, new Random(100));

    LearningCurve labeledOnlyCurve = new LearningCurve();
    LearningCurve labeledAndUnlabeledCurve = new LearningCurve();
    
    for(int fold = 0; fold < FOLDS; fold++) {
      
      labeledOnlyCurve.startNewFold();
      labeledAndUnlabeledCurve.startNewFold();
      
      Dataset labeled = new Dataset();
      Dataset nontest = splits[fold].getPoolSet();
      Dataset test = splits[fold].getTestSet();
      
      Dataset[] parts = nontest.split(MAXLABELED, nontest.size() - MAXLABELED);
      Dataset pool = parts[0]; // pool for labeling
      Dataset unlabeled = parts[1];
      
      while(true) {
        labeled.add(pool.popRandom(1, new Random(100)));
        labeled.setInstanceClassProbabilityDistribution(new HashSet<String>(dataset.getLabelAlphabet().getStrings()));

        double labeledOnlyAccuracy = EmAlgorithm.em(labeled, 
            unlabeled, 
            test, 
            dataset.getLabelAlphabet(), 
            dataset.getFeatureAlphabet(),
            0);
        double labeledAndUnlabeledAccuracy = EmAlgorithm.em(labeled, 
            unlabeled, 
            test, 
            dataset.getLabelAlphabet(), 
            dataset.getFeatureAlphabet(),
            EmAlgorithm.ITERATIONS);
        
        labeledOnlyCurve.add(labeled.size(), (float)labeledOnlyAccuracy);
        labeledAndUnlabeledCurve.add(labeled.size(), (float)labeledAndUnlabeledAccuracy);
        
        if(pool.size() == 0) {
          break;
        }
      }
    }
    
    labeledOnlyCurve.average();
    labeledAndUnlabeledCurve.average();
    labeledOnlyCurve.saveAveragedCurve(Constants.outputFileLabeledOnly);
    labeledAndUnlabeledCurve.saveAveragedCurve(Constants.outputFileLabeledAndUnlabeled);
    
    System.out.println("done!");
  }
}
