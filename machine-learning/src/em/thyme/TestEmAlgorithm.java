package em.thyme;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Random;

import data.Dataset;
import em.implementation.EmAlgorithm;

public class TestEmAlgorithm {

  public static final int STEP = 50;
  public static final int MAXLABELED = 3000;
  public static final int NUMUNLABELED = 300;
  public static final int ITERATIONS = 25;
  public static final int RNDSEED = 100;
  public static final String TRAINSET = "/Users/Dima/Boston/Data/Thyme/Vectors/EventTimeContains/train.txt";
  public static final String TESTSET = "/Users/Dima/Boston/Data/Thyme/Vectors/EventTimeContains/test.txt";

  public static void main(String[] args) throws IOException {
    for(int numberOfLabeledExamples = STEP; numberOfLabeledExamples < MAXLABELED; numberOfLabeledExamples += STEP) {
      double labeledOnlyAccuracy = testEm(numberOfLabeledExamples, 0);
      double labeledAndUnlabeledAccuracy = testEm(numberOfLabeledExamples, ITERATIONS);
      System.out.format("%d %.4f %.4f\n", numberOfLabeledExamples, labeledOnlyAccuracy, labeledAndUnlabeledAccuracy);
    }
  }

  public static double testEm(int numberOfLabeledExamples, int iterations) throws FileNotFoundException {

    Dataset pool = new Dataset();
    pool.loadCSVFile(TRAINSET);    
    pool.makeAlphabets();

    Dataset test = new Dataset();
    test.loadCSVFile(TESTSET);

    Dataset labeled = new Dataset();
    labeled.add(pool.popRandom(numberOfLabeledExamples, new Random(RNDSEED)));

    Dataset unlabeled = new Dataset();
    unlabeled.add(pool.popRandom(NUMUNLABELED, new Random(RNDSEED)));

    double accuracy = EmAlgorithm.runAndEvaluate(
        labeled, 
        unlabeled, 
        test, 
        pool.getLabelAlphabet(), 
        pool.getFeatureAlphabet(),
        iterations,
        1);

    return accuracy;
  }
}
