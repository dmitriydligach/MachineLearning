package em.thyme;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.HashSet;
import java.util.Random;

import data.Dataset;
import em.implementation.EmModel;

public class TestEmAlgorithm {

  public static final int STEP = 200;
  public static final int MAXLABELED = 10000;
  public static final int ITERATIONS = 25;
  public static final int RNDSEED = 100;
  public static final String TRAINSET = "/Users/Dima/Boston/Data/Thyme/Vectors/DocTimeRel/train.txt";
  public static final String TESTSET = "/Users/Dima/Boston/Data/Thyme/Vectors/DocTimeRel/test.txt";
  
  public static void main(String[] args) throws IOException {
    for(int numberOfLabeledExamples = STEP; numberOfLabeledExamples < MAXLABELED; numberOfLabeledExamples += STEP) {
      double labeledOnlyAccuracy = testEm(numberOfLabeledExamples, 0);
//      double labeledAndUnlabeledAccuracy = testEm(numberOfLabeledExamples, ITERATIONS);
      System.out.format("%d %.4f %.4f\n", numberOfLabeledExamples, labeledOnlyAccuracy, labeledOnlyAccuracy);
    }
  }
  
  public static double testEm(int numberOfLabeledExamples, int iterations) throws FileNotFoundException {
    
    Dataset pool = new Dataset();
    pool.loadCSVFile(TRAINSET);
    Dataset test = new Dataset();
    test.loadCSVFile(TESTSET);
    
    pool.makeAlphabets();
    pool.makeVectors();
    test.setAlphabets(pool.getLabelAlphabet(), pool.getFeatureAlphabet());
    test.makeVectors();
    
    Dataset labeled = new Dataset();
    labeled.add(pool.popRandom(numberOfLabeledExamples, new Random(RNDSEED)));
    labeled.setAlphabets(pool.getLabelAlphabet(), pool.getFeatureAlphabet());
    labeled.makeVectors();

//    pool.setInstanceClassProbabilityDistribution(new HashSet<String>(pool.getLabelAlphabet().getStrings()));
    labeled.setInstanceClassProbabilityDistribution(new HashSet<String>(pool.getLabelAlphabet().getStrings()));
    
    EmModel classifier = new EmModel(pool.getLabelAlphabet());
//    classifier.train(pool);
    classifier.train(labeled);
    double accuracy = classifier.test(test);

//  Dataset unlabeled = new Dataset();    
    
//      labeled.add(nontest.popRandom(numberOfLabeledExamples, new Random(100)));
//      unlabeled.add(nontest.popRandom(Constants.UNLABELED, new Random(100)));
//
//      double accuracy = EmAlgorithm.runAndEvaluate(
//          labeled, 
//          unlabeled, 
//          test, 
//          dataset.getLabelAlphabet(), 
//          dataset.getFeatureAlphabet(),
//          iterations);
//      cumulativeAccuracy = cumulativeAccuracy + accuracy;
    
    return accuracy;
  }
}
