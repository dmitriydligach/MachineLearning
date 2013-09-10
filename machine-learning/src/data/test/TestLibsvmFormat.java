package data.test;

import java.io.IOException;

import data.Dataset;

public class TestLibsvmFormat {

	public static void main(String[] args) throws IOException {
		
		Dataset instances = new Dataset();
		
		instances.loadCSVFile("/home/dima/i2b2/disease-activity/vectors/for-active/features.txt");
		System.out.println("done loading...");
		
		instances.makeAlphabets();
		instances.makeVectors();
		System.out.println("done making sparse vectors...");
		
		instances.writeLibsvmFile("/home/dima/i2b2/disease-activity/vectors/for-active/libsvm.txt");
		System.out.println("libsvm file written...");
	}
}
