package em.experiments;

import java.io.IOException;
import java.util.Random;

import cv.LearningCurve;
import data.Dataset;
import data.I2b2Dataset;
import data.Split;
import em.implementation.EmAlgorithm;

public class EmCurve {

  public static void main(String[] args) throws IOException {

    final int FOLDS = 10; // number of folds
    final int MAXLABELED = 100; // maximum number of labeled examples
    final int UNLABELED = 200;  // number of unlabeled examples
    final Random RANDOM = new Random(100);
        
    I2b2Dataset dataset = new I2b2Dataset();
    dataset.loadCSVFile(Constants.dataFile, Constants.labelFile);
    dataset.makeAlphabets();

    Split[] splits = dataset.split(FOLDS);

    LearningCurve labeledOnlyCurve = new LearningCurve();
    LearningCurve labeledAndUnlabeledCurve = new LearningCurve();
    
    for(int fold = 0; fold < FOLDS; fold++) {
      
      labeledOnlyCurve.startNewFold();
      labeledAndUnlabeledCurve.startNewFold();
      
      Dataset labeled = new Dataset();
      Dataset nontest = splits[fold].getPoolSet();
      Dataset test = splits[fold].getTestSet();
      
      Dataset[] parts = nontest.split(MAXLABELED, UNLABELED, RANDOM);
      Dataset pool = parts[0]; // pool for labeling of size MAXLABELED
      Dataset unlabeled = parts[1]; // unlabeled data (all nontest minus pool above)
      
      while(true) {
        labeled.add(pool.popRandom(1, RANDOM));

        // train using only labeled data
        double labeledOnlyAccuracy = EmAlgorithm.runAndEvaluate(
            labeled, 
            unlabeled, 
            test, 
            dataset.getLabelAlphabet(), 
            dataset.getFeatureAlphabet(),
            0);
        
        // train using both labeled and unlabeled data
        double labeledAndUnlabeledAccuracy = EmAlgorithm.runAndEvaluate(
            labeled, 
            unlabeled, 
            test, 
            dataset.getLabelAlphabet(), 
            dataset.getFeatureAlphabet(),
            Constants.ITERATIONS);
        
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
