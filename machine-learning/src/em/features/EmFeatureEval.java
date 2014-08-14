package em.features;

import java.io.IOException;
import java.util.HashSet;
import java.util.Random;

import semsup.eval.Constants;
import data.Alphabet;
import data.Dataset;
import data.I2b2Dataset;
import data.Split;
import em.implementation.EmModel;

public class EmFeatureEval {

  public static final int NUMLABELED = 10;
  public static final int NUMUNLABELED = 3000;
  public static final int FOLDS = 10; 
  public static final int ACTUALFOLDSTORUN = 1;
  public static final int ITERATIONS = 25;

  public static void main(String[] args) throws IOException {

    if(args.length < 1) {
      System.err.println("Please specify location of the properties file");
    } else {
      Constants.populate(args[0], false);  
    }

    I2b2Dataset dataset = new I2b2Dataset();
    I2b2Dataset unlabeled = new I2b2Dataset();
    dataset.loadCSVFile(Constants.cdData, Constants.cdLabels);
    unlabeled.loadFromCSVFile(Constants.cdData, Constants.cdLabels, NUMUNLABELED);
    // dataset.mapLabels(Constants.msSourceLabels, Constants.msTargetLabel);
    dataset.makeAlphabets();

    Split[] splits = dataset.split(FOLDS);
    double cumulAcc = 0;

    for(int fold = 1; fold < ACTUALFOLDSTORUN + 1; fold++) {
      Dataset labeled = new Dataset();
      Dataset nontest = splits[fold].getPoolSet();
      Dataset test = splits[fold].getTestSet();
      labeled.add(nontest.popRandom(NUMLABELED, new Random(Constants.rndSeed)));

      double accuracy = runAndEvaluate(
          labeled, 
          unlabeled,
          test, 
          dataset.getLabelAlphabet(), 
          dataset.getFeatureAlphabet(),
          ITERATIONS,
          1);

      cumulAcc = cumulAcc + accuracy;
    }

    System.out.format("cv accuracy: %.4f\n", cumulAcc / ACTUALFOLDSTORUN);
  }

  /**
   * Same EM algorithm but also prints feature weights.
   */
  public static double runAndEvaluate(
                          Dataset labeled, 
                          Dataset unlabeled, 
                          Dataset test, 
                          Alphabet labelAlphabet, 
                          Alphabet featureAlphabet,
                          int iterations,
                          double lambda) {

    labeled.setInstanceClassProbabilityDistribution(new HashSet<String>(labelAlphabet.getStrings()));
    labeled.setAlphabets(labelAlphabet, featureAlphabet);
    labeled.makeVectors();

    EmModel em = new EmModel(labelAlphabet, lambda);
    em.train(labeled);

    for(int iteration = 0; iteration < iterations; iteration++) {
      // E-step
      unlabeled.setAlphabets(labelAlphabet, featureAlphabet);
      unlabeled.makeVectors();
      em.label(unlabeled);

      // M-step
      Dataset labeledPlusUnlabeled = new Dataset(labeled.getInstances(), unlabeled.getInstances());
      labeledPlusUnlabeled.setAlphabets(labelAlphabet, featureAlphabet);
      labeledPlusUnlabeled.makeVectors();
      em.train(labeledPlusUnlabeled);

      // print feature weights
      if(iteration % ITERATIONS == 0) {
        System.out.println("\n* iteration " + iteration + "\n");
        try {
          NFoldCvFeatureEval.displayFeatureWeights(em, featureAlphabet);
        } catch (IOException e) {
          e.printStackTrace();
        }
      }
    }

    test.setAlphabets(labelAlphabet, featureAlphabet);
    test.makeVectors();
    double accuracy = em.test(test);

    return accuracy;
  }
}
