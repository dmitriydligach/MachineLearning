package classifier;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

import cc.mallet.classify.Classification;
import cc.mallet.classify.Classifier;
import cc.mallet.classify.MaxEntTrainer;
import cc.mallet.pipe.Noop;
import cc.mallet.pipe.Pipe;
import cc.mallet.share.upenn.MaxEntShell;
import cc.mallet.types.FeatureVector;
import cc.mallet.types.InstanceList;
import data.Dataset;
import data.Instance;

/**
 * This is a wrapper for the Mallet's maximum entropy classifier.
 * @author dmitriy dligach
 */
public class MalletMaxEntClassifier {

	public Classifier model;
	
	/**
	 * Train a maxent model.
	 */
	public void train(Dataset dataset) throws IOException {
		
		String[][] featureMatrix = toFeatureMatrix(dataset);
		String[] labels = toLabels(dataset);
		
		model = MaxEntShell.train(featureMatrix, labels, 1, null);
	}

	/**
	 * Train using instances
	 */
	public void train2(Dataset dataset) throws IOException {
		
		List<cc.mallet.types.Instance> malletInstances = new ArrayList<cc.mallet.types.Instance>();
		
		for(Instance instance : dataset.getInstances()) {
			cc.mallet.types.Instance malletInstance = toMalletInstance(instance, null);
			malletInstance.setTarget(instance.getLabel());
			malletInstances.add(malletInstance);
		}
		
		InstanceList instanceList = new InstanceList(new Noop());
		instanceList.addThruPipe(malletInstances.iterator());

		MaxEntTrainer maxEntTrainer = new MaxEntTrainer();
		model = maxEntTrainer.train(instanceList);
		
		// model = MaxEntShell.train(malletInstances.iterator(), 1, null);
	}

	/**
	 * Train using instances
	 */
	public void train3(Dataset dataset) throws IOException {
		
		cc.mallet.types.InstanceList instanceList = new cc.mallet.types.InstanceList(new Noop());
		
		for(Instance instance : dataset.getInstances()) {
			cc.mallet.types.Instance malletInstance = toMalletInstance(instance, null);
			malletInstance.setTarget(instance.getLabel());
			instanceList.add(malletInstance);
		}
	
		MaxEntTrainer maxEntTrainer = new MaxEntTrainer();
		model = maxEntTrainer.train(instanceList);
	}
	
	/**
	 * Train again
	 */
	public void train4(Dataset dataset) throws IOException {
		
		ArrayList<Pipe> pipeList = new ArrayList<Pipe>();
		pipeList.add(new Noop());
		// SerialPipes serialPipe = new SerialPipes(pipeList);
		
		// InstanceList instanceList = new InstanceList(serialPipe);
	}
	
	/**
	 * Classify test set. Return accuracy.
	 */
	public double test(Dataset dataset) {
		
		int correct = 0;
		
		for(Instance instance : dataset.getInstances()) {
			// cc.mallet.types.Instance malletInstance = toMalletInstance(instance, model.getAlphabet());
			cc.mallet.types.Instance malletInstance = toMalletInstance(instance, null); 
			
			Classification classification = model.classify(malletInstance);
			String prediction = classification.getLabeling().getBestLabel().toString();

			if(prediction.equals(instance.getLabel())) {
				correct++;
			}
		}
		
		return (double) correct / dataset.size();
	}
	
	/**
	 * Convert an instance to the Mallet Instance object.
	 */
	public static cc.mallet.types.Instance toMalletInstance(data.Instance instance, cc.mallet.types.Alphabet malletAlphabet) {
				
		int[] indices = new int[instance.getVector().size()];
		double[] values = new double[instance.getVector().size()];
		
		int i = 0;
		for(int dimension : instance.getVector().keySet()) {
			indices[i] = dimension;
			values[i] = 1; // instance.getDimensionValue(dimension);
			i++;
		}
		
		FeatureVector featureVector = new FeatureVector(malletAlphabet, indices, values);
		return new cc.mallet.types.Instance(featureVector, null, null, null);
	}
	
	/**
	 * Convert a dataset object into a two dimensional array
	 * in which each row (of unequal length) represents the "on" features.
	 */
	public static String[][] toFeatureMatrix(Dataset dataset) {

		String[][] featureMatrix = new String[dataset.size()][];
		
		for(int i = 0; i < dataset.size(); i++) {
			data.Instance instance = dataset.getInstances().get(i);
			Set<String> features = instance.getFeatures().keySet();
			featureMatrix[i] = new String[features.size()];
			featureMatrix[i] = features.toArray(new String[1]);
		}

		return featureMatrix;
	}

	/**
	 * Return an array that consists of the labels of the instances
	 * in the given dataset.
	 */
	public static String[] toLabels(Dataset dataset) {
		
		String[] labels = new String[dataset.size()];
		
		for(int i = 0; i < dataset.size(); i++) {
			data.Instance instance = dataset.getInstances().get(i);
			labels[i] = instance.getLabel();
		}

		return labels;
	}
	
	/**
	 * Convert a data.Aplhabet object to the Mallet Alphabet object. 
	 */
	public cc.mallet.types.Alphabet toMalletAlphabet(data.Alphabet alphabet) {
		
		String[] features = alphabet.getStrings().toArray(new String[1]);
		cc.mallet.types.Alphabet malletAlphabet = new cc.mallet.types.Alphabet(features);
		
		return malletAlphabet;
	}
}
