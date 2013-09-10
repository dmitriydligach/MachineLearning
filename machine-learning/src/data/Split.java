package data;

import java.util.List;

public class Split {
	
	private Dataset trainSet;
	private Dataset testSet;
	
	public Split() {
		trainSet = new Dataset();
		testSet = new Dataset();
	}
	
	/**
	 * Construct from two lists of instances. 
	 */
	public Split(List<Instance> trainInstances, List<Instance> testInstances) {
		
		trainSet = new Dataset(trainInstances);
		testSet = new Dataset(testInstances);
	}
	
	public Dataset getPoolSet() {
	  return trainSet;
  }
	
	public void setTrainSet(Dataset trainSet) {
	  this.trainSet = trainSet;
  }
	
	public Dataset getTestSet() {
	  return testSet;
  }

	public void setTestSet(Dataset testSet) {
	  this.testSet = testSet;
  }
}
