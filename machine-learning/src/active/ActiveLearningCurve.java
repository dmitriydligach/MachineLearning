package active;

import java.io.IOException;
import java.util.Random;

import classifier.NaiveBayesClassifier;
import cv.LearningCurve;
import data.Dataset;
import data.Instance;
import data.Split;

public class ActiveLearningCurve {
	
	public static void main(String[] args) throws IOException {

		Dataset dataset = new Dataset();
		dataset.loadCSVFile(Constants.inputFile);
		dataset.makeAlphabets();
		
		Split[] splits = dataset.split(Constants.totalFolds);
		LearningCurve learningCurve = new LearningCurve();
		
		for(int fold = 0; fold < Constants.totalFolds; fold++) {
			learningCurve.startNewFold();
			
			Dataset pool = splits[fold].getPoolSet();
			Dataset test = splits[fold].getTestSet();
			Dataset train = new Dataset();

			// seed (should be the same for active learning and random sampling)
			train.add(pool.popRandom(Constants.seedSize, new Random(Constants.rndSeedForSeeding)));
			
			while(true) {
				NaiveBayesClassifier classifier = new NaiveBayesClassifier(dataset.getLabelAlphabet());

				// train a model
				train.makeAlphabets();
				train.makeVectors();
				classifier.train(train);
				
				// get this model's performance on the training set
				test.setAlphabets(train.getLabelAlphabet(), train.getFeatureAlphabet());
				test.makeVectors();
				double accuracy = classifier.test(test);
			
				// record the performance at this size of the training set
				learningCurve.add(train.size(), (float)accuracy);

				// was last pool instance added to train set during previous iteration?
				if(pool.size() == 0) {
					break; 
				}
				
				// select next instance using the current model
				pool.setAlphabets(train.getLabelAlphabet(), train.getFeatureAlphabet());
				pool.makeVectors();
				Instance instance = classifier.getMostUncertainInstance(pool); 

				// add selected instance to the training set
				train.add(instance);
			}
		}
		
		learningCurve.average();
		learningCurve.saveAveragedCurve(Constants.outputFileActive);
		System.out.println("active learning done!");
	}
}
