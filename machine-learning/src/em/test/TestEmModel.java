package em.test;

import java.io.IOException;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

import data.Dataset;
import data.I2b2Dataset;
import data.Split;
import em.implementation.EmModel;

public class TestEmModel {
	
  public static Set<String> msSourceLabels = new HashSet<String>(Arrays.asList("2", "3", "4", "5"));
  public static String msTargetLabel = "2";
  public static Set<String> t2dSourceLabels = new HashSet<String>(Arrays.asList("\"possible\""));
  public static String t2dTargetLabel = "\"no\"";
  
	public static void main(String[] args) throws IOException {

		I2b2Dataset dataset = new I2b2Dataset();
		dataset.loadCSVFile(Constants.DATAFILE, Constants.LABELFILE);
		dataset.mapLabels(t2dSourceLabels, t2dTargetLabel);
		dataset.makeAlphabets(); // need label alphabet to init NB classifier
		
		Split[] splits = dataset.split(Constants.FOLDS);
		
		double cumulativeAccuracy = 0;
		for(int fold = 0; fold < Constants.FOLDS; fold++) {
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
			
			cumulativeAccuracy = cumulativeAccuracy + accuracy;
		}
		
		double accuracy = cumulativeAccuracy / Constants.FOLDS;
		System.out.format("%d-fold cv accuracy: %.4f\n", Constants.FOLDS, accuracy);
	}
}
