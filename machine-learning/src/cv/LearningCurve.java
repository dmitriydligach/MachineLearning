package cv;

import java.io.BufferedWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.TreeMap;

import com.google.common.collect.HashMultiset;
import com.google.common.collect.Multiset;

public class LearningCurve {
	
	// current fold number (may not need this)
	int currentFold;
	
	// Each List element contains results from a single fold
	// TreeMap stores accuracy for each training set size
	private List<TreeMap<Integer, Float>> learningCurves;
	
	// Number of experiments that was conducted for each train set size
	private Multiset<Integer> normalizationCounts;
	
	// learning curve produced by averaging at each train set size
	private TreeMap<Integer, Float> averagedLearningCurve;
	
	public LearningCurve() {
		currentFold = -1;
		learningCurves = new ArrayList<TreeMap<Integer, Float>>();
		normalizationCounts = HashMultiset.create();
		averagedLearningCurve = new TreeMap<Integer, Float>();
	}
	
	public void startNewFold() {
		currentFold++;
		learningCurves.add(new TreeMap<Integer, Float>());
	}
	
	public void add(int size, float accuracy) {
		learningCurves.get(currentFold).put(size, accuracy);
		normalizationCounts.add(size);
	}
	
	public void average() {
		
		// sum up accuracies at each size
		for(TreeMap<Integer, Float> learningCurve : learningCurves) {
			for(int size : learningCurve.keySet()) {
				if(averagedLearningCurve.containsKey(size)) {
					averagedLearningCurve.put(size, averagedLearningCurve.get(size) + learningCurve.get(size));
				} else {
					averagedLearningCurve.put(size, learningCurve.get(size));
				}
			}
		}
		
		// divided by the total number of times each size was seen
		for(int size : averagedLearningCurve.keySet()) {
			averagedLearningCurve.put(size, averagedLearningCurve.get(size) / normalizationCounts.count(size));
		}
	}
	
	public void saveAveragedCurve(String fileName) throws IOException {
		
		BufferedWriter bufferedWriter = utils.Misc.getWriter(fileName);
		
		for(int size : averagedLearningCurve.keySet()) {
			bufferedWriter.write(size + " " + averagedLearningCurve.get(size) + "\n");
		}
		
		bufferedWriter.close();
	}
}
