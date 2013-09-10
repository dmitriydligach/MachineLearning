package data.test;

import data.Alphabet;

public class TestAlphabet {

	public static void main(String[] args) {

		Alphabet alphabet = new Alphabet();
	
		alphabet.add("one");
		alphabet.add("two");
		alphabet.add("three");
		alphabet.add("one");
		alphabet.add("two");
		alphabet.add("four");
		
		System.out.println();
		System.out.println(alphabet.getString(2));
		System.out.println(alphabet.getIndex("two"));
	}
}
