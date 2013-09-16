package classifier.test;

import java.io.FileNotFoundException;

import classifier.NaiveBayesClassifier;
import data.Dataset;
import data.I2b2Dataset;
import data.Split;

public class TestI2b2Format {
	
	public static void main(String[] args) throws FileNotFoundException {
		
		final int N = 5; // number of folds

		I2b2Dataset dataset = new I2b2Dataset();

		dataset.loadCSVFile("/home/dima/active/t2d/data/data.txt", "/home/dima/active/t2d/data/labels.txt");
		System.out.println("total instances loaded: " + dataset.size());
		dataset.makeAlphabets();
		Split[] splits = dataset.split(N);
		
		double cumulativeAccuracy = 0;
		for(int fold = 0; fold < N; fold++) {
			Dataset trainSet = splits[fold].getPoolSet();
			Dataset testSet = splits[fold].getTestSet();
			
			trainSet.makeAlphabets();
			trainSet.makeVectors();

			testSet.setAlphabets(trainSet.getLabelAlphabet(), trainSet.getFeatureAlphabet());
			testSet.makeVectors();
			
			NaiveBayesClassifier classifier = new NaiveBayesClassifier(dataset.getLabelAlphabet());
			classifier.train(trainSet);

			double accuracy = classifier.test(testSet, "\"yes\"");

			cumulativeAccuracy = cumulativeAccuracy + accuracy;
		}
		
		double accuracy = cumulativeAccuracy / N;
		System.out.println("average accuracy:\t" + accuracy);
	}
}
