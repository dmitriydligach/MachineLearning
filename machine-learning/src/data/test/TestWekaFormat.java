package data.test;

import java.io.IOException;

import data.Dataset;

public class TestWekaFormat {

	public static void main(String[] args) throws IOException {
		
		Dataset instances = new Dataset();
		
		instances.loadCSVFile("/home/dima/i2b2/disease-activity/vectors/for-active/features.txt");
		instances.makeAlphabets();
		instances.makeVectors();
		instances.writeWekaFile("/home/dima/temp/data.arff");
	}
}
