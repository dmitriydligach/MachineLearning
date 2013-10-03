package em.test;

import java.io.IOException;
import java.util.HashSet;

import data.Dataset;
import data.I2b2Dataset;
import data.Split;
import em.implementation.EmModel;

public class TestEmModel {
	
	public static void main(String[] args) throws IOException {

		final int FOLDS = 10; // number of folds
	  final String DATAFILE = "/home/dima/active/ms/data/data.txt";
	  final String LABELFILE = "/home/dima/active/ms/data/labels.txt";
		
		I2b2Dataset dataset = new I2b2Dataset();
		dataset.loadCSVFile(DATAFILE, LABELFILE);
		dataset.makeAlphabets(); // need label alphabet to init NB classifier
		
		Split[] splits = dataset.split(FOLDS);
		
		double cumulativeAccuracy = 0;
		for(int fold = 0; fold < FOLDS; fold++) {
			Dataset trainSet = splits[fold].getPoolSet();
			Dataset testSet = splits[fold].getTestSet();
			
			trainSet.makeAlphabets();
			trainSet.makeVectors();

			testSet.setAlphabets(trainSet.getLabelAlphabet(), trainSet.getFeatureAlphabet());
			testSet.makeVectors();

			trainSet.setInstanceClassProbabilityDistribution(new HashSet<String>(dataset.getLabelAlphabet().getStrings()));
			EmModel classifier = new EmModel(dataset.getLabelAlphabet());
			
			classifier.train(trainSet);
			double accuracy = classifier.test(testSet);
			
			cumulativeAccuracy = cumulativeAccuracy + accuracy;
		}
		
		double accuracy = cumulativeAccuracy / FOLDS;
		System.out.format("%d-fold cv accuracy: %.4f\n", FOLDS, accuracy);
	}
}
