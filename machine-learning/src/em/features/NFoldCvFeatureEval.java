package em.features;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Map;

import semsup.eval.Constants;
import utils.UmlsLookup;

import com.google.common.base.Function;
import com.google.common.base.Functions;
import com.google.common.collect.Ordering;

import data.Alphabet;
import data.Dataset;
import data.I2b2Dataset;
import data.Split;
import em.implementation.EmModel;

public class NFoldCvFeatureEval {

  public static final int FOLDS = 10; 
  public static final int ACTUALFOLDSTORUN = 10;
  public static final int FEATURESTOPRINT = 50;
  
  public static void main(String[] args) throws IOException {

    if(args.length < 1) {
      System.err.println("Please specify location of the properties file");
    } else {
      Constants.populate(args[0], false);  
    }

    I2b2Dataset dataset = new I2b2Dataset();
    dataset.loadCSVFile(Constants.t2dData, Constants.t2dLabels);
    dataset.mapLabels(Constants.t2dSourceLabels, Constants.t2dTargetLabel);
    dataset.makeAlphabets();

    Split[] splits = dataset.split(FOLDS);
    double cumulAcc = 0;

    // run just one fold and look at feature weights
    for(int fold = 0; fold < ACTUALFOLDSTORUN; fold++) {
      Dataset trainSet = splits[fold].getPoolSet();
      Dataset testSet = splits[fold].getTestSet();

      trainSet.makeAlphabets();
      trainSet.makeVectors();
      testSet.setAlphabets(trainSet.getLabelAlphabet(), trainSet.getFeatureAlphabet());
      testSet.makeVectors();

      trainSet.setInstanceClassProbabilityDistribution(new HashSet<String>(dataset.getLabelAlphabet().getStrings()));
      EmModel classifier = new EmModel(dataset.getLabelAlphabet(), 1.0);
      classifier.train(trainSet);
      double accuracy = classifier.test(testSet);
      cumulAcc = cumulAcc + accuracy;

      displayFeatureWeights(classifier, dataset.getFeatureAlphabet(), FEATURESTOPRINT);
      System.out.println();
    }

    System.out.format("accuracy: %.4f\n", cumulAcc / ACTUALFOLDSTORUN);
  }

  /**
   * Print feature weights. 
   * Last argument specifies the maximum number of features to print.
   */
  public static void displayFeatureWeights(EmModel model, Alphabet featureAlphabet, int maxFeatures) throws IOException {

    Map<String, Double> featureWeights = model.computeFeatureWeights(featureAlphabet);
    List<String> featureNamesSortedByWeight = new ArrayList<String>(featureWeights.keySet());
    Function<String, Double> getValue = Functions.forMap(featureWeights);
    Collections.sort(featureNamesSortedByWeight, Ordering.natural().reverse().onResultOf(getValue));

    UmlsLookup umlsLookup = new UmlsLookup("/Users/dima/Boston/Data/Umls/rxnorm-snomedct.txt");
    int featuresToInclude = featureWeights.size() < maxFeatures ? featureWeights.size() : maxFeatures;

    for(int featureNum = 0; featureNum < featuresToInclude; featureNum++) {
      String feature = featureNamesSortedByWeight.get(featureNum);
      String capitalizedNegationRemoved = feature.replace("c", "C").replace("-", "");
      String text = umlsLookup.getTerm(capitalizedNegationRemoved);
      if(text == null) {
        text = "n/a";
      }
      int charsToPrint = text.length() > 50 ? 50 : text.length(); 
      System.out.format("%12s %-50s %.4f\n", feature, text.substring(0, charsToPrint), featureWeights.get(feature));
    }
  }
}
