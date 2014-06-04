package em.thyme;

import java.io.IOException;
import java.util.HashSet;

import data.Dataset;
import em.implementation.EmModel;

public class TestEmModel {

  public static void main(String[] args) throws IOException {

    Dataset trainSet = new Dataset();
    trainSet.loadCSVFile("/Users/Dima/Boston/Output/DocTime/doc-time-rel-train.txt");
    trainSet.makeAlphabets(); // need label alphabet to init NB classifie

    Dataset testSet = new Dataset();
    testSet.loadCSVFile("/Users/Dima/Boston/Output/DocTime/doc-time-rel-test.txt");
    testSet.makeAlphabets(); // need label alphabet to init NB classifie

    trainSet.makeAlphabets();
    trainSet.makeVectors();
    testSet.setAlphabets(trainSet.getLabelAlphabet(), trainSet.getFeatureAlphabet());
    testSet.makeVectors();
    trainSet.setInstanceClassProbabilityDistribution(new HashSet<String>(trainSet.getLabelAlphabet().getStrings()));
    EmModel classifier = new EmModel(trainSet.getLabelAlphabet());
    classifier.train(trainSet);
    double accuracy = classifier.test(testSet);

    System.out.format("accuracy: %.4f\n", accuracy);
  }
}
