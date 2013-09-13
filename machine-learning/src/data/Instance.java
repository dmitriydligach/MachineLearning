package data;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;

import com.google.common.base.Joiner;

public class Instance {

	// class label
	private String label;

	// Feature name-value pairs
	private Map<String, Float> features;

	// sparse vector representation (e.g. "22->4, 28->1, ...")
	private Map<Integer, Float> vector;

	// probability distribution over classes
	private Map<String, Float> labels;
	
	public Instance() {
		label = "";
		features = new HashMap<String, Float>();
		vector = new TreeMap<Integer, Float>();
		labels = new HashMap<String, Float>();
	}

	/**
	 * Create from an instance. 
	 */
	public Instance(Instance sourceInstance) {

		label = sourceInstance.getLabel();
		features = new HashMap<String, Float>();
		vector = new TreeMap<Integer, Float>(); 
		labels = sourceInstance.getClassProbabilities(); // TODO: figure out where to set this
		
		// features can be removed by feature selection, so need deep copy
		// features = sourceInstance.getFeatures(); 
		for(String feature : sourceInstance.getFeatures().keySet()) {
			features.put(feature, sourceInstance.getFeatureValue(feature));
		}
	}
	
	public Map<String, Float> getFeatures() {
		return features;
	}
	
	public Map<Integer, Float> getVector() {
		return vector;
	}
	
	public Map<String, Float> getClassProbabilities() {
	  return labels;
	}
	
	public float getClassProbability(String label) {
		return labels.get(label);
	}
	
	/**
	 * Hard class assignement using this instance's label.
	 */
	public void setClassProbabilities(Set<String> uniqueLabels) {
		
		for(String l : uniqueLabels) {
			if(l.equals(label)) {
				labels.put(l, 1.0f);
			} else {
				labels.put(l, 0.0f);
			}
		}
	}
	
	/**
	 * Soft class assignments (e.g. obtained from a classifier).
	 */
	public void setLabelProbabilityDistribution(Map<String, Float> classDistribution) {
		
		for(String key : classDistribution.keySet()) {
			labels.put(key, classDistribution.get(key));
		}
	}
	
	public void addFeature(String name, float value) {
		features.put(name, value);
	}
	
	public Float getFeatureValue(String name) {
		return features.get(name);
	}
	
	public Float getDimensionValue(int dimension) {
		return vector.get(dimension);
	}
	
	/**
	 * Sparse vectors may need to be regenerated.
	 */
	public void resetVector() {
		
	}
	
	public void setVector(int dimension, float value) {
		vector.put(dimension, value);
	}
	
	public String getLabel() {
	  return label;
  }
	
	public void setLabel(String label) {
	  this.label = label;
  }
	
	/**
	 * Represent sparse vector as string using the following format:
	 * 
	 * <dimension><value separator><value><feature separator> ...
	 * e.g. 8:7.0 11:2.0 15:1.0 16:2.0 ...
	 */
	public String getVectorAsString(String valueSeparator, String featureSeparator) {
		
		List<String> stringRepresentation = new ArrayList<String>();
		
		for(Integer dimension : vector.keySet()) {
			String nameValuePair = String.format("%d%s%s", dimension, valueSeparator, vector.get(dimension));
			stringRepresentation.add(nameValuePair);
		}
		
		Joiner joiner = Joiner.on(featureSeparator);
		return joiner.join(stringRepresentation);
	}
	
	/**
	 * Sum up the values of all features.
	 */
	public float getTotalMass() {
		
		float mass = 0;
		for(float value : features.values()) {
			mass += value;
		}

		return mass;
	}
}
