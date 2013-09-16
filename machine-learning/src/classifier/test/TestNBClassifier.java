package classifier.test;

import java.io.IOException;

import classifier.NaiveBayesClassifier;
import data.Dataset;
import data.Split;

public class TestNBClassifier {
	
	public static void main(String[] args) throws IOException {

		final int N = 5; // number of folds
		
		Dataset dataset = new Dataset();
		dataset.loadCSVFile("/home/dima/i2b2/disease-activity/vectors/for-active/features.txt");
		dataset.makeAlphabets(); // need label alphabet to init NB classifier
		
		Split[] splits = dataset.split(N);
		System.out.println("done splitting");
		
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
			double accuracy = classifier.test(testSet);
			
			cumulativeAccuracy = cumulativeAccuracy + accuracy;
		}
		
		double accuracy = cumulativeAccuracy / N;
		System.out.println("average accuracy:\t" + accuracy);
	}
}
