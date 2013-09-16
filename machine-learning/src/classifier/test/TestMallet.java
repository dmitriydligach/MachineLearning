package classifier.test;

import java.io.IOException;

import classifier.MalletMaxEntClassifier;
import data.Dataset;
import data.Split;

public class TestMallet {

	public static void main(String[] args) throws IOException {

		final int N = 5; // number of folds
		
		Dataset dataset = new Dataset();
		dataset.loadCSVFile("/home/dima/i2b2/disease-activity/vectors/for-active/features.txt");
		
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
			
			MalletMaxEntClassifier maxEntClassifier = new MalletMaxEntClassifier();
			maxEntClassifier.train2(trainSet);
			double accuracy = maxEntClassifier.test(testSet);

			cumulativeAccuracy = cumulativeAccuracy + accuracy;
		}
		
		double accuracy = cumulativeAccuracy / N;
		System.out.println("average accuracy:\t" + accuracy);
	}
}
