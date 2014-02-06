package gibbs;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;

import utils.Misc;

import com.google.common.base.Function;
import com.google.common.base.Functions;
import com.google.common.collect.Ordering;

import data.Alphabet;
import data.Dataset;
import data.Instance;

/**
 * Implements a multinomial naive bayes classifier.
 * 
 * @author dmitriy dligach
 *
 */
public class Model {

	// number of classes
	protected int numClasses;
	// number of words (features)
	protected int numWords;
	// number of examples in training data
	protected int numInstances;
	
	// word (feature) counts for each class
	protected double[][] wordCounts;
	// number of examples for each class
	protected int[] labelCounts;
	// total word (feature) count in each class
	private int[] totalClassWords;

	// p(w|c) for all classes 
	protected double[][] theta;
	// p(c) for all classes
	protected double[] priors;

	// map labels to ints and ints to labels
	Alphabet labelAlphabet;
	
	/** 
	 * Set the label alphabet here. This cannot be done in initialize()
	 * using the dataset's alphabet because sometimes the training data
	 * may not have all the labels that exist in the test data.
	 */
	public Model(Alphabet labelAlphabet) 
	{
		this.labelAlphabet = labelAlphabet;
	}
	
	/**
	 * Initialize various counts needed for training.
	 * Assumes alphabet and vectors generated for this set.
	 */
	protected void initialize(Dataset dataset) {
		
		numClasses = labelAlphabet.size();
		numWords = dataset.getNumberOfDimensions();
		numInstances = dataset.size();
		
		wordCounts = new double[numClasses][numWords];
		labelCounts = new int[numClasses];
		totalClassWords = new int[numClasses];
		
		priors = new double[numClasses];
		theta = new double[numClasses][numWords];
	}

	/**
	 * Train a multinomial naive bayes model using a dataset.
	 */
	public void train(Dataset dataset) {

		initialize(dataset);
		
		// compute various useful counts
		computeWordCounts(dataset);
		computeLabelCounts(dataset);
		computeTotalClassWords(dataset);
		
		// compute naive bayes parameters
		computeTheta();
		computePriors();
	}
		
	/**
	 * Classify instances in a dataset. Return accuracy.
	 */
	public double test(Dataset dataset) {
		
		int correct = 0;
		int total = 0;
		
		for(Instance instance : dataset.getInstances()) {
			int prediction = classify(instance);
			if(prediction == labelAlphabet.getIndex(instance.getLabel())) {
				correct++;
			}
			total++;
		}

		return (double) correct / total;
	}

	/**
	 * Classify instances in a dataset. Return F1 for the given label.
	 */
	public double test(Dataset dataset, String label) {
		
		int correctLabelPredictions = 0; // number of times the label was predicted correctly
		int totalLabelPredictions = 0;   // number of times the classifier predicted the label
		int totalLabelInstances = 0;     // number of instances of the label in the dataset
		
		for(Instance instance : dataset.getInstances()) {
			int prediction = classify(instance);
			
			if(prediction == labelAlphabet.getIndex(label)) {
				totalLabelPredictions++;
				if(prediction == labelAlphabet.getIndex(instance.getLabel())) {
					correctLabelPredictions++;
				}
			}
			
			if(label.equals(instance.getLabel())) {
				totalLabelInstances++;
			}
		}
		
		double precision = (double) correctLabelPredictions / totalLabelPredictions;
		double recall = (double) correctLabelPredictions / totalLabelInstances;
			
		return (2 * precision * recall) / (precision + recall);
	}
		
	
	/**
	 * Classify a document using a multinomial naive bayes model. 
	 */
	public int classify(Instance instance) {
		double[] classLogProbs = getUnnormalizedClassLogProbs(instance);
		return Misc.getIndexOfLargestElement(classLogProbs);
	}

	/**
	 * Find the most problematic instance for the current model. 
	 * Remove this instance from the dataset and return it.
	 */
	public Instance getMostUncertainInstance(Dataset dataset) {

		// key: instance index, value: uncertainty margin
		HashMap<Integer, Double> scoredInstances = new HashMap<Integer, Double>();

		for(int instanceIndex = 0; instanceIndex < dataset.size(); instanceIndex++) {
			Double instanceScore = computeUncertainty(dataset.getInstance(instanceIndex));
			scoredInstances.put(instanceIndex, instanceScore);
		}
		
		// sort by margin size; small margin first (e.g. p(c1) = 0.49, p(c2) = 0.51)
		List<Integer> sortedKeys = new ArrayList<Integer>(scoredInstances.keySet());
    Function<Integer, Double> getValue = Functions.forMap(scoredInstances);
    Collections.sort(sortedKeys, Ordering.natural().onResultOf(getValue));

    // remove and return the instance with smallest margin
    return dataset.removeInstance(sortedKeys.get(0));
	}
	
	/**
	 * Compute uncertainty margin for an instance, which is
	 * |p(most probable class) - p(second most probable class)|
	 */
	public Double computeUncertainty(Instance instance) {
		// TODO: generalize to n classes
		
		double[] logSum = getUnnormalizedClassLogProbs(instance);
		double[] p = logToProb(logSum[0], logSum[1]);
		return Math.abs(p[0] - p[1]);
	}
	
	/**
	 * Convert two unnormalized log probs to probabilities.
	 */
	public double[] logToProb(double logP0, double logP1) {

		double[] p = new double[2];
		
		// odds0 is p0/p1 or p0 / (1 - p0)
		double odds0 = Math.pow(10, logP0 - logP1);

		p[0] = odds0 / (1 + odds0);
		p[1] = 1 / (1 + odds0);
		
		return p;
	}
	
	/**
	 * Calculate for each class:
	 * 
	 * p(c|w_0, ..., w_n-1) ~ p(c)p(w_0|c)p(w_1|c)...p(w_n-1|c)
	 *   where w_0, ..., w_n-1 are word tokens in the document
	 *   
	 * Calculations are done in log space. I.e. we need to calculate:
	 * log(p(c)p(w_0|c)...p(w_n-1|c)) = log(p(c)) + log(p(w_0|c)) + ... + log(p(w_n-1|c))
	 * 
	 * OOV words (i.e. words not seen during training) are currently ignored.
	 * 
	 * TODO: For the inner loop, iterate over words that exist in this vector.
	 * For some of them (OOV words), p(w|c) will be unknown (ignore them).
	 */
	public double[] getUnnormalizedClassLogProbs(Instance instance) {
		
		double[] logSum = new double[numClasses];

		for(int label = 0; label < numClasses; label++) {
			logSum[label] = Math.log10(priors[label]);

			// iterate over words that were seen during training
			for(int word = 0; word < numWords; word++) {
				Float wordCount = instance.getDimensionValue(word);
				if(wordCount == null) {
					// this instance does not contain the current word; just go on to
					// the next word (i.e. pretend the wordCount for this word is zero)
					continue;
				}
				
				double wordProb = theta[label][word];        
				double prob = Math.pow(wordProb, wordCount); 
				logSum[label] += Math.log10(prob);
			}
		}
		
		return logSum;
	}

	/**
	 * Compute p(w|c) parameters.
	 * 
	 * Smoothing is needed because feature alphabet is computed based
	 * on the entire dataset (i.e. all classes). If no smoothing is done,
	 * for some words, class probability p(w|c) will be zero because some 
	 * words are only seen in the instances of one class. For classification,
	 * we need to ensure that no p(w|c) in the model is zero.
	 * 
	 * During classification, there will be OOV words (i.e. words that were not
	 * seen during training). These words will be just ignored.
	 */
	public void computeTheta() {
		
		for(int label = 0; label < numClasses; label++) {
			for(int word = 0; word < numWords; word++) {
				theta[label][word] = (wordCounts[label][word] + 1) / (totalClassWords[label] + numWords);
			}
		}
	}
	
	/**
	 * Compute p(c) parameters.
	 */
	public void computePriors() {
		
		for(int label = 0; label < numClasses; label++) {
			priors[label] = (double) labelCounts[label] / numInstances;
		}
	}
	
	/**
	 * Compute feature frequency n x m matrix, where
	 * n: number of classes, m: number of dimensions
	 * 
	 * TODO: Make more efficient by changing the inner loop
	 * to iterate over feature names (later converted to dimensions)
	 * instead of all dimensions.
	 */
	public void computeWordCounts(Dataset dataset) {
		
		for(Instance instance : dataset.getInstances()) {
			for(int index = 0; index < numWords; index++) {
				int label = labelAlphabet.getIndex(instance.getLabel());
				Float value = instance.getDimensionValue(index); 
				if(value != null) {
					wordCounts[label][index] += value;
				}
			}
		}
	}
	
	/**
	 * For each label, compute the number of examples that have that label.
	 */
	public void computeLabelCounts(Dataset dataset) {
		
		for(Instance instance : dataset.getInstances()) {
			int label = labelAlphabet.getIndex(instance.getLabel());
			labelCounts[label]++;
		}
	}
	
	/**
	 * Compute total number of words in each class.
	 */
	public void computeTotalClassWords(Dataset dataset) {

		for(Instance instance : dataset.getInstances()) {
			int label = labelAlphabet.getIndex(instance.getLabel());
			totalClassWords[label] += instance.getTotalMass();
		}
	}
	
	/**
	 * Display the thetas for each class (sorted).
	 */
	public void printModel(Alphabet featureAlphabet) {
		
		List<HashMap<String, Double>> allClassThetas = new ArrayList<HashMap<String, Double>>();
		
		for(int label = 0; label < numClasses; label++) {
			HashMap<String, Double> classThetas = new HashMap<String, Double>();
			
			for(int word = 0; word < numWords; word++) {
				String string = featureAlphabet.getString(word);
				classThetas.put(string, theta[label][word]);
			}
			
	    List<String> sortedKeys = new ArrayList<String>(classThetas.keySet());
	    Function<String, Double> getValue = Functions.forMap(classThetas);
	    Collections.sort(sortedKeys, Ordering.natural().reverse().onResultOf(getValue));

	    System.out.format("p(%s) = %f\n\n", labelAlphabet.getString(label), priors[label]);
	    
	    for(int i = 0; i < 5; i++) {
	    	String key = sortedKeys.get(i);
	    	System.out.format("p(%s|%s) = %f\n", key, labelAlphabet.getString(label), classThetas.get(key));
			}
	    System.out.println();
	    
	    allClassThetas.add(classThetas);
		}
	}
	
	/**
	 * Do feature contribution analysis. Use p(w|c1) / p(w|c2) for ranking features.
	 * Features that get the scores close to 1, do not contribute much.
	 */
	public void featureLogOdds(List<HashMap<String, Double>> allClassThetas) {
		
		HashMap<String, Double> ratios = new HashMap<String, Double>();
		
		for(String feature : allClassThetas.get(0).keySet()) {
			double ratio = 0.0;
			if(allClassThetas.get(0).get(feature) > allClassThetas.get(1).get(feature)) {
				ratio = allClassThetas.get(0).get(feature) / allClassThetas.get(1).get(feature);
			} else {
				ratio = allClassThetas.get(1).get(feature) / allClassThetas.get(0).get(feature);
			}
			ratios.put(feature, Math.abs(ratio)); 
		}
		
    List<String> sortedKeys = new ArrayList<String>(ratios.keySet());
    Function<String, Double> getValue = Functions.forMap(ratios);
    Collections.sort(sortedKeys, Ordering.natural().reverse().onResultOf(getValue));

    for(String key : sortedKeys) {
    	System.out.format("score(%s) = %f\n", key, ratios.get(key));
		}
	}
}