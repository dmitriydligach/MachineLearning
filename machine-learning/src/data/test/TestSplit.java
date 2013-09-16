package data.test;

import java.io.IOException;

import data.Dataset;
import data.Split;

public class TestSplit {

	public static void main(String[] args) throws IOException {
		
		Dataset instances = new Dataset();
		
		instances.loadCSVFile("/home/dima/i2b2/disease-activity/vectors/token-features/features.txt");

		Split[] splits = instances.split(5);
		
		splits[2].getPoolSet().writeLibsvmFile("/home/dima/temp/train.txt");
		splits[2].getTestSet().writeLibsvmFile("/home/dima/temp/test.txt");
	}
}
