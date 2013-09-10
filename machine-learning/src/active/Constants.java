package active;

public class Constants {
	
	public static final String inputFile = "/home/dima/i2b2/disease-activity/vectors/for-active/features.txt";
	public static final String outputFileRandom = "/home/dima/temp/random.txt";
	public static final String outputFileActive = "/home/dima/temp/active.txt";
	
	public static final int totalFolds = 10; 
	public static final int seedSize = 25;
	
	public static final int rndSeedForSplitting = 0;
	public static final int rndSeedForSeeding = 10; // 5 works; 10 works really well; 500 does not work
}
