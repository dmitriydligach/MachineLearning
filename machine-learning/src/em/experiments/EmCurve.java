package em.experiments;

import java.io.IOException;
import java.util.HashSet;
import java.util.Random;

import cv.LearningCurve;
import data.Dataset;
import data.I2b2Dataset;
import data.Split;
import em.implementation.EmApi;

public class EmCurve {

  public static void main(String[] args) throws IOException {

    final int FOLDS = 10; // number of folds
    final int ITERATIONS = 50; // number of iterations
        
    I2b2Dataset dataset = new I2b2Dataset();
    dataset.loadCSVFile("/home/dima/active/ibd/data/data.txt", "/home/dima/active/ibd/data/labels-cd.txt");
    dataset.makeAlphabets();

    Split[] splits = dataset.split(FOLDS, new Random(100));
    LearningCurve learningCurve = new LearningCurve();
    
    for(int fold = 0; fold < FOLDS; fold++) {
      learningCurve.startNewFold();
      
      Dataset labeled = new Dataset();
      Dataset nontest = splits[fold].getPoolSet();
      Dataset test = splits[fold].getTestSet();
      Split[] parts = nontest.split(2, new Random(100));
      Dataset pool = parts[0].getPoolSet(); // pool for labeling
      Dataset unlabeled = parts[0].getTestSet();

      while(true) {
        labeled.add(pool.popRandom(1, new Random(100)));
        labeled.setInstanceClassProbabilityDistribution(new HashSet<String>(dataset.getLabelAlphabet().getStrings()));
        
        // are these needed?
        labeled.setAlphabets(dataset.getLabelAlphabet(), dataset.getFeatureAlphabet());
        labeled.makeVectors();

        double accuracy = EmApi.em(labeled, 
            unlabeled, 
            test, 
            dataset.getLabelAlphabet(), 
            dataset.getFeatureAlphabet(),
            ITERATIONS);
        
        learningCurve.add(labeled.size(), (float)accuracy);
        
        if(pool.size() == 0) {
          break;
        }
      }
    }
    
    learningCurve.average();
    learningCurve.saveAveragedCurve(Constants.outputFileActive);
    System.out.println("done!");
  }
}
