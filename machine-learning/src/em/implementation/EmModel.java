package em.implementation;

import java.math.BigDecimal;
import java.math.BigInteger;
import java.math.RoundingMode;
import java.util.HashMap;
import java.util.Map;

import utils.Misc;
import data.Alphabet;
import data.Dataset;
import data.Instance;

/**
 * Implements a multinomial naive bayes classifier as described in:
 * 
 * Text Classification from Labeled and Unlabeled Documents using EM (2000)
 * Kamal Nigam, Andrew Kachites Mccallum, Sebastian Thrun, and Tom Mitchell
 * 
 * @author dmitriy dligach
 *
 */
public class EmModel {

  // weight of unlabeled data
  public final static double LAMBDA = 1.0;
  
  // number of classes in training and test data
  protected int numClasses;
  // number of words (features) in training data
  protected int numWords;
  // p(w|c) for all classes 
  protected double[][] theta;
  // p(c) for all classes
  protected double[] priors;
  // map labels to ints and ints to labels
  protected Alphabet labelAlphabet;
  
  /** 
   * Set the label alphabet here. This cannot be done in initialize()
   * using the dataset's alphabet because sometimes the training data
   * may not have all the labels that exist in the test data.
   */
	public EmModel(Alphabet labelAlphabet) {
	  this.labelAlphabet = labelAlphabet;
	}
	
  /**
   * Initialize various counts and data structures 
   * needed for training and train a model using a dataset. 
   * Assume alphabet and vectors have been generated for the dataset. 
   */
  public void train(Dataset dataset) {
    numClasses = labelAlphabet.size();
    numWords = dataset.getNumberOfDimensions();
    priors = new double[numClasses];
    theta = new double[numClasses][numWords];
    computeTheta(dataset);
    computePriors(dataset);
  }
    
  /**
   * Calculate p(w|c) for all words in training data 
   * and for each class. Based on equation 13 in the paper.
   */
  private void computeTheta(Dataset dataset) {
    double[][] wordCounts = computeWordCounts(dataset);
    double[] totalWordCountInClass = new double[numClasses];    
    for(int label = 0; label < numClasses; label++) {
      for(int word = 0; word < numWords; word++) {
        totalWordCountInClass[label] += wordCounts[label][word];        
      }
    }
    // calculcate p(w|c) for each word in each class
    for(int label = 0; label < numClasses; label++) {
      for(int word = 0; word < numWords; word++) {
        theta[label][word] = (1 + wordCounts[label][word]) / (numWords + totalWordCountInClass[label]);
        assert !Double.isNaN(theta[label][word]);
        assert !Double.isInfinite(theta[label][word]);
      }
    }
  }
  
  /**
   * Compute how many times each word occurs in each class.
   * Values could be fractional since class membership is a probability.  
   */
  private double[][] computeWordCounts(Dataset dataset) {
    double[][] wordCounts = new double[numClasses][numWords];
    for(int wordIndex = 0; wordIndex < numWords; wordIndex++) {
      for(int classIndex = 0; classIndex < numClasses; classIndex++) {
        wordCounts[classIndex][wordIndex] = 0;
        for(Instance instance : dataset.getInstances()) {
          double lambda = (instance.getLabel() == null ? LAMBDA : 1.0);
          String label = labelAlphabet.getString(classIndex);
          Float wordCount = instance.getDimensionValue(wordIndex); // null if count = 0 for this word
          if(wordCount != null) {
            wordCounts[classIndex][wordIndex] += lambda * wordCount * instance.getClassProbability(label); 
          }
        }
      }
    }
    return wordCounts;
  }
  
  /**
   * Compute p(c) for each class. Based on on Equation 14 in the paper.
   */
  private void computePriors(Dataset dataset) {
    int numLabeled = 0;
    int numUnlabeled = 0;
    for(Instance instance : dataset.getInstances()) {
      if(instance.getLabel() == null) {
        numUnlabeled++;
      } else {
        numLabeled++;
      }
    }
    for(int classIndex = 0; classIndex < numClasses; classIndex++) {
      double sum = 0;
      for(Instance instance : dataset.getInstances()) {
        double lambda = (instance.getLabel() == null ? LAMBDA : 1.0);
        String label = labelAlphabet.getString(classIndex);
        sum += lambda * instance.getClassProbability(label);
      }
      priors[classIndex] = (1 + sum) / (numClasses + numLabeled + LAMBDA * numUnlabeled);
      assert !Double.isNaN(priors[classIndex]);
      assert !Double.isInfinite(priors[classIndex]);
    }
  }

  /**
   * Classify instances in a dataset. Return accuracy.
   */
  public double test(Dataset dataset) {
    int correct = 0;
    int total = 0;
    for(Instance instance : dataset.getInstances()) {
      // classify by picking class with largest unnormalized log probability
      int prediction = Misc.getIndexOfLargestElement(getUnnormalizedClassLogProbs(instance));
      if(prediction == labelAlphabet.getIndex(instance.getLabel())) {
        correct++;
      }
      total++;
    }
    return (double) correct / total;
  }
  
	/**
   * Classify instances in a dataset. 
   * Set probability distribution over classes for each instance.
   */
  public void label(Dataset dataset) {
    for(Instance instance : dataset.getInstances()) {
      double[] logSum = getUnnormalizedClassLogProbs(instance);
      double[] p = logToProb(logSum);
      Map<String, Float> labelProbabilityDistribution = new HashMap<String, Float>();
      for(int label = 0; label < numClasses; label++) {
        labelProbabilityDistribution.put(labelAlphabet.getString(label), (float)p[label]);
      }
      instance.setLabelProbabilityDistribution(labelProbabilityDistribution); 
    }
  }
  
  /**
   * A copy of method above that's different in that
   * Instances' "label" field is overwritten with prediction.
   * Eventually, this version should the replace the one above.
   */
  public void label2(Dataset dataset) {
    for(Instance instance : dataset.getInstances()) {
      double[] logSum = getUnnormalizedClassLogProbs(instance);
      double[] p = logToProb(logSum);
      
      Map<String, Float> labelProbabilityDistribution = new HashMap<String, Float>();
      for(int label = 0; label < numClasses; label++) {
        labelProbabilityDistribution.put(labelAlphabet.getString(label), (float)p[label]);
      }
      instance.setLabelProbabilityDistribution(labelProbabilityDistribution); 

      String prediction = labelAlphabet.getString(Misc.getIndexOfLargestElement(p));
      instance.setLabel(prediction);
    }
  }

  /**
   * Calculate data log-likelihood given current model. 
   * Based on equation 9 in the paper.
   */
  public double getDataLogLikelihood(Dataset labeled, Dataset unlabeled, int numDecimalPlaces) {
    double dataLogLikelihood = 0.0; 
    for(Instance instance : unlabeled.getInstances()) {
      double[] classLogProbs = getUnnormalizedClassLogProbs(instance);
      double instanceProbability = 0.0;
      for(int label = 0; label < numClasses; label++) {
        instanceProbability = instanceProbability + Math.pow(10, classLogProbs[label]);
      }
      if(instanceProbability != 0) {
        dataLogLikelihood = dataLogLikelihood + Math.log10(instanceProbability);
      }
    }
    for(Instance instance : labeled.getInstances()) {
      double[] classLogProbs = getUnnormalizedClassLogProbs(instance);
      double instanceProbability = classLogProbs[labelAlphabet.getIndex(instance.getLabel())];
      dataLogLikelihood = dataLogLikelihood + instanceProbability;
    }
    return (double) Math.round(dataLogLikelihood * 10 * numDecimalPlaces) / (10 * numDecimalPlaces);
  }
  
  /**
   * Calculate data log-likelihood given current model. 
   * Based on equation 9 in the paper.
   */
  public double getDataLogLikelihood2(Dataset labeled, Dataset unlabeled, int numDecimalPlaces) {
    double dataLogLikelihood = 0.0; 
    for(Instance instance : unlabeled.getInstances()) {
      double[] classLogProbs = getUnnormalizedClassLogProbs(instance);
      BigDecimal instanceProbability = new BigDecimal(0);
      for(int label = 0; label < numClasses; label++) {
        instanceProbability = instanceProbability.add(powerOfTen(classLogProbs[label]));
      }
      dataLogLikelihood = dataLogLikelihood + log(instanceProbability, 2);
    }
    for(Instance instance : labeled.getInstances()) {
      double[] classLogProbs = getUnnormalizedClassLogProbs(instance);
      double instanceProbability = classLogProbs[labelAlphabet.getIndex(instance.getLabel())];
      dataLogLikelihood = dataLogLikelihood + instanceProbability;
    }
    return (double) Math.round(dataLogLikelihood * 10 * numDecimalPlaces) / (10 * numDecimalPlaces);
  }
  
  /**
   * Calculate log of a big decimal.
   * log10(unsc_val * 10^-scale) = log10(unsc_val) + log10(10^-scale) = log10(unsc_val) - scale
   */
  public static double log(BigDecimal argument, int numDecimalPlaces) {
    // log10(unsc_val * 10^-scale) = log10(unsc_val) + log10(10^-scale) = log10(unsc_val) - scale
    BigInteger unscaledValue = argument.unscaledValue();
    double unscaledValueRounded = (double) Math.round(unscaledValue.doubleValue() * 10 * numDecimalPlaces) / (10 * numDecimalPlaces);
    int scale = argument.scale();
    return Math.log10(unscaledValueRounded) - scale;
  }
  
  /**
   * Calculate data likelihood given current model. 
   * Based on equation 8 in the paper.
   */
  public BigDecimal getDataLikelihood(Dataset labeled, Dataset unlabeled, int numDecimalPlaces) {
    BigDecimal dataLikelihood = new BigDecimal(0);
    for(Instance instance : unlabeled.getInstances()) {
      double[] classLogProbs = getUnnormalizedClassLogProbs(instance); // log[p(class, instance)] for all classess
      BigDecimal instanceProbability = new BigDecimal(0);              // sum out the class to get p(instance)
      for(int label = 0; label < numClasses; label++) {
        instanceProbability = instanceProbability.add(powerOfTen(classLogProbs[label]));
      }
      dataLikelihood = dataLikelihood.multiply(instanceProbability);
    }
    for(Instance instance : labeled.getInstances()) {
      double[] classLogProbs = getUnnormalizedClassLogProbs(instance);
      BigDecimal instanceProbability = powerOfTen(classLogProbs[labelAlphabet.getIndex(instance.getLabel())]);
      dataLikelihood = dataLikelihood.multiply(instanceProbability);  
    }
    return dataLikelihood;
  }
  
	 /**
   * Calculate for each class:
   * 
   * p(c|w_0, ..., w_n-1) ~ p(c)p(w_0|c)p(w_1|c)...p(w_n-1|c)
   * where w_0, ..., w_n-1 are words in the document in positions w_0, ..., w_n-1.
   *   
   * Calculations are done in log space. I.e. we need to calculate:
   * log[p(c)p(w_0|c)...p(w_n-1|c)] = log[p(c)] + log[p(w_0|c)] + ... + log[p(w_n-1|c)]
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
        logSum[label] += wordCount * Math.log10(theta[label][word]);
        assert !Double.isNaN(logSum[label]);
        assert !Double.isInfinite(logSum[label]);
      }
    }
    return logSum;
  }
	
  /**
   * Calculate 10^exponent when y is very small and fractional.
   * Basically split exponent into its integer i and fractional f parts.
   * I.e. 10^exponent = 10^(i + f) = 10^i * 10^f
   */
  private static BigDecimal powerOfTen(double exponent) {
    BigDecimal exponentAsBigDecimal = new BigDecimal(String.valueOf(exponent));
    BigDecimal integerPart = new BigDecimal(exponentAsBigDecimal.intValue());
    BigDecimal fractionalPart = exponentAsBigDecimal.subtract(integerPart);
    BigDecimal tenToIntegerPart = new BigDecimal(BigInteger.valueOf(10), -1 * integerPart.intValue());
    BigDecimal tenToFractionalPart = new BigDecimal(Math.pow(10, fractionalPart.doubleValue()));
    BigDecimal result = tenToIntegerPart.multiply(tenToFractionalPart);
    return result;
  }
    
  /**
   * Convert unnormalized log probabilities to probabilities for each class.
   */
  public double[] logToProb(double[] unnormalizedClassLogProbs) {
    double[] probs = new double[unnormalizedClassLogProbs.length];
    // unnormalized probabilities are often very small, e.g. 10^-802.345
    // which causes an underflow, so need to use big decimal instead
    BigDecimal[] unnormalizedClassProbs = new BigDecimal[unnormalizedClassLogProbs.length];
    // we have log(p(c)p(w_0|c)...p(w_n-1|c)) for each class
    // compute unnormalized probabilities as 10^(p(c)p(w_0|c)...p(w_n-1|c))
    for(int label = 0; label < numClasses; label++) {
      unnormalizedClassProbs[label] = powerOfTen(unnormalizedClassLogProbs[label]);
    }
    // compute the normalization constant
    BigDecimal normalizer = new BigDecimal(0);
    for(int label = 0; label < numClasses; label++) {
      normalizer = normalizer.add(unnormalizedClassProbs[label]);
    }
    for(int label = 0; label < numClasses; label++) {
      probs[label] = unnormalizedClassProbs[label].divide(normalizer, RoundingMode.HALF_UP).doubleValue();
    }
    return probs;
  }
  
  /**
   * Convert two unnormalized log probs to probabilities.
   */
  @Deprecated
  public double[] logToProb(double logP0, double logP1) {
    double[] p = new double[2];
    // difference logP0 - logP1 larger than a few hundred causes an overflow
    // when evaluating 10 ^ (logP0 - logP1); however the difference of that 
    // size means that p0 is many orders of magnitude larger than p1;
    // so we can just assume that p0 is one and p1 is zero
    // e.g. logP0 = -1000, logP1 = -2000, logP0 - logP1 = 1000;
    // which means that p0 / p1 is 10^1000, i.e. p0 >> p1
    // the underflow is not an issue; e.g. 10^-1000 is just zero
    // TODO: can use Double.MAX_VALUE instead of 300
    final int MAXDIFFERENCE = 300;
    if((logP0 - logP1) > MAXDIFFERENCE) {
      p[0] = 1.0;
      p[1] = 0.0;
      
      return p;
    }
    // odds0 is p0/p1 or p0 / (1 - p0)
    double odds0 = Math.pow(10, logP0 - logP1);
    p[0] = odds0 / (1 + odds0);
    p[1] = 1 / (1 + odds0);
    assert !Double.isNaN(p[0]);
    assert !Double.isNaN(p[1]);
    assert !Double.isInfinite(p[0]);
    assert !Double.isInfinite(p[1]);
    return p;
  }
}
